import os
import glob
import math
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed()

if torch.cuda.is_available():
    available_gpu_count = torch.cuda.device_count()
    if available_gpu_count >= 8:
        device = torch.device("cuda:0")
        gpu_ids = [0, 1]
        logging.info(f"Using GPUs: {gpu_ids} for training.")
    else:
        device = torch.device("cuda:1")
        gpu_ids = None
        logging.warning("Insufficient available GPUs, using single GPU training.")
else:
    device = torch.device("cpu")
    gpu_ids = None
    logging.warning("No GPU detected, using CPU.")

class ProteinDataset(Dataset):
    def __init__(self, features, labels, eids):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.eids = eids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.eids[idx]

def std_name(name: str) -> str:
    return name.strip().lower()

def find_common_healthy_controls(disease_label_files, position_path):
    logging.info("Finding common healthy controls across all diseases...")
    pos_df = pd.read_csv(position_path)
    valid_eids = set(pos_df[pos_df['p54_i0'] == 0]['eid'].unique())
    common_healthy_eids = valid_eids.copy()
    for label_file in disease_label_files:
        disease_name = os.path.basename(label_file).split('_labels.csv')[0]
        labels_df = pd.read_csv(label_file)
        healthy_eids = set(labels_df[labels_df['label'] == 0]['eid'].unique())
        common_healthy_eids = common_healthy_eids.intersection(healthy_eids)
        logging.info(f"Disease {disease_name}: {len(healthy_eids)} healthy controls, common healthy: {len(common_healthy_eids)}")
    logging.info(f"Found {len(common_healthy_eids)} common healthy controls across all diseases")
    return list(common_healthy_eids)

def load_and_prepare_data(features_path, labels_path, position_path, protein_corr_path, protein_seq_path, common_healthy_eids=None):
    logging.info(f"Loading expression data: {features_path}")
    expr_df = pd.read_csv(features_path)
    logging.info(f"Loading disease labels: {labels_path}")
    labels_df = pd.read_csv(labels_path)
    logging.info("Loading participant position information and filtering where p54_i0 == 0")
    pos_df = pd.read_csv(position_path)
    pos_df = pos_df[pos_df['p54_i0'] == 0]
    expr_df = expr_df.drop_duplicates(subset=['eid'])
    labels_df = labels_df.drop_duplicates(subset=['eid'])
    pos_df = pos_df.drop_duplicates(subset=['eid'])
    if common_healthy_eids is not None:
        cases_df = labels_df[labels_df['label'] == 1]
        healthy_df = labels_df[labels_df['eid'].isin(common_healthy_eids)]
        healthy_df = healthy_df[healthy_df['label'] == 0]
        labels_df = pd.concat([cases_df, healthy_df], ignore_index=True)
        logging.info(f"Using {len(cases_df)} cases and {len(healthy_df)} common healthy controls")
    df = pd.merge(expr_df, labels_df, on='eid')
    df = pd.merge(df, pos_df[['eid']], on='eid')
    df = df.dropna()
    class_counts = df['label'].value_counts()
    logging.info(f"Class distribution - Cases: {class_counts.get(1, 0)}, Controls: {class_counts.get(0, 0)}")
    logging.info(f"Loading bias matrix: {protein_corr_path}")
    bias_df = pd.read_csv(protein_corr_path, index_col=0)
    bias_df = bias_df.fillna(0)
    logging.info(f"Loading protein sequence feature representations: {protein_seq_path}")
    with np.load(protein_seq_path) as data:
        protein_features = {key: data[key] for key in data.files}
    logging.info(f"Number of protein sequence features: {len(protein_features)}")
    all_expr_proteins = [col for col in df.columns if col not in ['eid', 'label']]
    expr_name_mapping = {std_name(col): col for col in all_expr_proteins}
    bias_names = {std_name(x) for x in bias_df.index}
    seq_names = {std_name(x) for x in protein_features.keys()}
    common_names_std = set(expr_name_mapping.keys()) & bias_names & seq_names
    if len(common_names_std) == 0:
        raise ValueError("No common proteins found in expression data, bias matrix and sequence features!")
    common_proteins = [col for col in all_expr_proteins if std_name(col) in common_names_std]
    logging.info(f"Number of common proteins: {len(common_proteins)}")
    df_filtered = df[['eid', 'label'] + common_proteins].copy()
    X = df_filtered[common_proteins].values.astype(np.float32)
    y = df_filtered['label'].values.astype(np.float32)
    eids = df_filtered['eid'].values
    bias_mapping = {std_name(x): x for x in bias_df.index}
    common_bias = [bias_mapping[std_name(prot)] for prot in common_proteins if std_name(prot) in bias_mapping]
    bias_df = bias_df.loc[common_bias, common_bias]
    bias_matrix = bias_df.values.astype(np.float32)
    bias_matrix = np.pad(bias_matrix, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    seq_mapping = {std_name(x): x for x in protein_features.keys()}
    seq_features_arr = np.stack([protein_features[seq_mapping[std_name(prot)]] for prot in common_proteins])
    return X, y, eids, bias_matrix, seq_features_arr, common_proteins

def create_balanced_sampler(labels):
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def pad_features(X, segment_size, total_features):
    num_samples, current_features = X.shape
    pad_size = total_features - current_features
    if pad_size > 0:
        X_padded = np.pad(X, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
    else:
        X_padded = X
    return X_padded

def split_bias_matrix_and_seq_features(bias_matrix, seq_features, segment_size):
    bias_core = bias_matrix[1:, 1:]
    total_features = bias_core.shape[0]
    num_segments = int(np.ceil(total_features / segment_size))
    bias_matrices = []
    seq_features_list = []
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size
        if end_idx > total_features:
            pad_size = end_idx - total_features
            bias_seg = bias_core[start_idx:total_features, start_idx:total_features]
            bias_seg = np.pad(bias_seg, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0)
        else:
            bias_seg = bias_core[start_idx:end_idx, start_idx:end_idx]
        bias_seg = np.pad(bias_seg, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        bias_matrices.append(bias_seg)
        seq_seg = seq_features[start_idx:min(end_idx, total_features), :]
        if seq_seg.shape[0] < segment_size:
            pad_size = segment_size - seq_seg.shape[0]
            seq_seg = np.pad(seq_seg, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        seq_features_list.append(seq_seg.astype(np.float32))
    return bias_matrices, seq_features_list, num_segments

class Tokenizer(nn.Module):
    def __init__(self, d_numerical: int, d_token: int, seq_features: torch.Tensor = None):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))
        self.bias_param = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias_param)
        if seq_features is not None:
            self.register_buffer('seq_features_buf', seq_features)
            if seq_features.shape[1] != d_token:
                self.seq_proj = nn.Linear(seq_features.shape[1], d_token)
            else:
                self.seq_proj = None
        else:
            self.seq_features_buf = None
            self.seq_proj = None

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        batch_size = x_num.size(0)
        cls_tokens = torch.ones(batch_size, 1, device=x_num.device)
        x_num = torch.cat([cls_tokens, x_num], dim=1)
        tokens = self.weight.unsqueeze(0) * x_num.unsqueeze(-1) + self.bias_param.unsqueeze(0)
        if self.seq_features_buf is not None:
            seq_feat = self.seq_features_buf
            if self.seq_proj is not None:
                seq_feat = self.seq_proj(seq_feat)
            tokens_without_cls = tokens[:, 1:, :] * seq_feat.unsqueeze(0)
            tokens = torch.cat([tokens[:, :1, :], tokens_without_cls], dim=1)
        return tokens

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, bias_matrix=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        if bias_matrix is not None:
            self.register_buffer('bias_matrix_buf', torch.from_numpy(bias_matrix).float())
        else:
            self.bias_matrix_buf = None
        self.attn_weights = None

    def forward(self, query, key, value, **kwargs):
        seq_len, batch_size, _ = query.size()
        def linear_projection(x, linear):
            x = linear(x)
            x = x.transpose(0, 1).contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim)
            return x.transpose(1, 2)
        Q = linear_projection(query, self.q_linear)
        K = linear_projection(key, self.k_linear)
        V = linear_projection(value, self.v_linear)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        if self.bias_matrix_buf is not None:
            bias = self.bias_matrix_buf.unsqueeze(0).unsqueeze(0)
            bias = bias.to(scores.device)
            scores = scores + bias
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        self.attn_weights = attn.detach()
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(context)
        return out.transpose(0, 1), attn

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, bias_matrix):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=True,
            bias_matrix=bias_matrix
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class SegmentTransformer(nn.Module):
    def __init__(self, segment_size, d_token, n_heads, n_layers, dim_feedforward, dropout, bias_matrix, seq_features):
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical=segment_size, d_token=d_token,
                                   seq_features=torch.tensor(seq_features, dtype=torch.float32)
                                   if seq_features is not None else None)
        seq_len = segment_size + 1
        assert bias_matrix.shape == (seq_len, seq_len), "Bias matrix shape error."
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bias_matrix=bias_matrix
            ) for _ in range(n_layers)
        ])

    def forward(self, x, return_attention=False):
        tokens = self.tokenizer(x)
        tokens = tokens.transpose(0, 1)
        attn_list = []
        for layer in self.layers:
            tokens = layer(tokens)
            attn_list.append(layer.self_attn.attn_weights)
        cls_token = tokens[0, :, :]
        if return_attention:
            return cls_token, attn_list
        else:
            return cls_token

class TransformerModel(nn.Module):
    def __init__(self,
                 n_features: int,
                 segment_size: int,
                 num_segments: int,
                 d_token: int,
                 n_heads: int,
                 n_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 bias_matrices: list,
                 seq_features_list: list):
        super().__init__()
        self.segment_size = segment_size
        self.num_segments = num_segments
        self.segment_transformers = nn.ModuleList([
            SegmentTransformer(
                segment_size=segment_size,
                d_token=d_token,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bias_matrix=bias_matrices[i],
                seq_features=seq_features_list[i]
            ) for i in range(num_segments)
        ])
        total_d_token = d_token * num_segments
        self.classifier = nn.Linear(total_d_token, 1)

    def forward(self, x, return_attention=False, return_features=False):
        segment_outputs = []
        attn_outputs = []
        for i in range(self.num_segments):
            start_idx = i * self.segment_size
            end_idx = (i + 1) * self.segment_size
            seg_input = x[:, start_idx:end_idx]
            if return_attention:
                seg_out, seg_attn_list = self.segment_transformers[i](seg_input, return_attention=True)
                attn_outputs.append(seg_attn_list)
            else:
                seg_out = self.segment_transformers[i](seg_input)
            segment_outputs.append(seg_out)
        concatenated_output = torch.cat(segment_outputs, dim=-1)
        logits = self.classifier(concatenated_output).squeeze(1)
        if return_attention:
            return logits, concatenated_output, attn_outputs
        elif return_features:
            return logits, concatenated_output
        else:
            return logits

def model_forward(model, X_batch, return_attention=False, return_features=False):
    if isinstance(model, nn.DataParallel):
        if return_attention:
            return model.module(X_batch, return_attention=True, return_features=False)
        elif return_features:
            return model.module(X_batch, return_attention=False, return_features=True)
        else:
            return model(X_batch)
    else:
        if return_attention:
            return model(X_batch, return_attention=True, return_features=False)
        elif return_features:
            return model(X_batch, return_attention=False, return_features=True)
        else:
            return model(X_batch)

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_targets = []
    for X_batch, y_batch, _ in tqdm(dataloader, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model_forward(model, X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
        all_preds.append(torch.sigmoid(logits).detach().cpu())
        all_targets.append(y_batch.detach().cpu())
    epoch_loss /= len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    binary_preds = (all_preds >= 0.5).float()
    epoch_auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())
    epoch_acc = accuracy_score(all_targets.numpy(), binary_preds.numpy())
    epoch_prec = precision_score(all_targets.numpy(), binary_preds.numpy(), zero_division=0)
    epoch_rec = recall_score(all_targets.numpy(), binary_preds.numpy(), zero_division=0)
    epoch_f1 = f1_score(all_targets.numpy(), binary_preds.numpy(), zero_division=0)
    return epoch_loss, epoch_auc, epoch_acc, epoch_prec, epoch_rec, epoch_f1

def eval_epoch(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch, _ in tqdm(dataloader, desc="Evaluation", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model_forward(model, X_batch)
            loss = criterion(logits, y_batch)
            epoch_loss += loss.item() * X_batch.size(0)
            all_preds.append(torch.sigmoid(logits).cpu())
            all_targets.append(y_batch.cpu())
    epoch_loss /= len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    binary_preds = (all_preds >= 0.5).float()
    epoch_auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())
    epoch_acc = accuracy_score(all_targets.numpy(), binary_preds.numpy())
    epoch_prec = precision_score(all_targets.numpy(), binary_preds.numpy(), zero_division=0)
    epoch_rec = recall_score(all_targets.numpy(), binary_preds.numpy(), zero_division=0)
    epoch_f1 = f1_score(all_targets.numpy(), binary_preds.numpy(), zero_division=0)
    return epoch_loss, epoch_auc, epoch_acc, epoch_prec, epoch_rec, epoch_f1

def main():
    features_path = 'proteomics_data.csv'
    position_path = 'position.csv'
    protein_corr_path = 'protein_correlation_matrix.csv'
    protein_seq_path = 'global_representations.npz'
    disease_label_pattern = 'label/*_labels.csv'
    disease_files = glob.glob(disease_label_pattern)
    if not disease_files:
        logging.error("No disease label files detected!")
        return
    common_healthy_eids = find_common_healthy_controls(disease_files, position_path)
    segment_size = 512
    metrics_summary_list = []
    for label_file in disease_files:
        disease_name = os.path.basename(label_file).split('_labels.csv')[0]
        logging.info(f"Start training disease: {disease_name}")
        X, y, eids, bias_matrix_full, seq_features_full, common_proteins = load_and_prepare_data(
            features_path, label_file, position_path, protein_corr_path, protein_seq_path,
            common_healthy_eids=common_healthy_eids)
        logging.info(f"Expression data shape: {X.shape}, Number of labels: {len(y)}")
        total_proteins = len(common_proteins)
        num_segments = int(np.ceil(total_proteins / segment_size))
        total_features = num_segments * segment_size
        X_padded = pad_features(X, segment_size, total_features)
        logging.info(f"Padded data shape: {X_padded.shape}")
        bias_matrices, seq_features_list, num_segments_calc = split_bias_matrix_and_seq_features(
            bias_matrix_full, seq_features_full, segment_size)
        if num_segments != num_segments_calc:
            raise ValueError("Mismatch in number of segments!")
        X_train, X_val, y_train, y_val, eids_train, eids_val = train_test_split(
            X_padded, y, eids, test_size=0.2, random_state=42, stratify=y)
        logging.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        logging.info(f"Training class distribution - Cases: {sum(y_train == 1)}, Controls: {sum(y_train == 0)}")
        logging.info(f"Validation class distribution - Cases: {sum(y_val == 1)}, Controls: {sum(y_val == 0)}")
        train_dataset = ProteinDataset(X_train, y_train, eids_train)
        val_dataset = ProteinDataset(X_val, y_val, eids_val)
        full_dataset = ProteinDataset(X_padded, y, eids)
        n_heads = 8
        d_token = 512
        n_layers = 6
        dim_forward = 2048
        dropout = 0.1
        lr = 2e-5
        batch_size = 32
        model = TransformerModel(
            n_features=X_train.shape[1],
            segment_size=segment_size,
            num_segments=num_segments,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_forward,
            dropout=dropout,
            bias_matrices=bias_matrices,
            seq_features_list=seq_features_list
        ).to(device)
        if gpu_ids is not None and len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
            logging.info(f"Using GPUs: {gpu_ids} for training.")
        else:
            logging.info("Using single GPU/CPU for training.")
        pos = sum(train_dataset.y.numpy() == 1)
        neg = sum(train_dataset.y.numpy() == 0)
        pos_weight = torch.tensor(neg / pos, dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_sampler = create_balanced_sampler(y_train)
        epochs = 15
        best_auc = 0
        best_state = None
        best_metrics = None
        metrics_history = []
        for epoch in range(epochs):
            logging.info(f"[{disease_name}] Epoch {epoch + 1}/{epochs}")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                      num_workers=16, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=16, pin_memory=True)
            train_loss, train_auc, train_acc, train_prec, train_rec, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer)
            val_loss, val_auc, val_acc, val_prec, val_rec, val_f1 = eval_epoch(
                model, val_loader, criterion)
            logging.info(f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
            logging.info(f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss, 'train_auc': train_auc, 'train_acc': train_acc,
                'train_prec': train_prec, 'train_rec': train_rec, 'train_f1': train_f1,
                'val_loss': val_loss, 'val_auc': val_auc, 'val_acc': val_acc,
                'val_prec': val_prec, 'val_rec': val_rec, 'val_f1': val_f1
            })
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()
                best_metrics = {
                    "Disease": disease_name.replace('_', ' ').title(),
                    "Epoch": epoch + 1,
                    "Train_Loss": train_loss,
                    "Train_AUC": train_auc,
                    "Train_Accuracy": train_acc,
                    "Train_Precision": train_prec,
                    "Train_Recall": train_rec,
                    "Train_F1": train_f1,
                    "Val_Loss": val_loss,
                    "Val_AUC": val_auc,
                    "Val_Accuracy": val_acc,
                    "Val_Precision": val_prec,
                    "Val_Recall": val_rec,
                    "Val_F1": val_f1
                }
                torch.save(best_state, f'best_model_{disease_name}.pth')
                logging.info(f"Saved current best model to best_model_{disease_name}.pth")
        metrics_df = pd.DataFrame(metrics_history)
        metrics_df.to_csv(f'metrics_history_{disease_name}.csv', index=False)
        logging.info(f"Detailed metrics history saved to metrics_history_{disease_name}.csv")
        if best_metrics is not None:
            metrics_summary_list.append(best_metrics)
        model.load_state_dict(torch.load(f'best_model_{disease_name}.pth', map_location=device))
        model.eval()
        all_logits = []
        all_features = []
        all_targets = []
        all_eids = []
        attn_sum_list = [None] * num_segments
        total_count = 0
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=16, pin_memory=True)
        with torch.no_grad():
            for X_batch, y_batch, eid_batch in tqdm(full_loader, desc="Prediction and attention extraction"):
                X_batch = X_batch.to(device)
                logits, features, attn_outputs = model_forward(model, X_batch, return_attention=True)
                all_logits.append(torch.sigmoid(logits).detach().cpu())
                all_features.append(features.detach().cpu())
                all_targets.append(y_batch.detach().cpu())
                all_eids.extend(eid_batch)
                for seg_idx, seg_attn_list in enumerate(attn_outputs):
                    attn_layers = torch.stack(seg_attn_list, dim=0)
                    attn_avg = attn_layers.mean(dim=0)
                    attn_cls = attn_avg[:, :, 0, 1:]
                    avg_attn = attn_cls.mean(dim=0).mean(dim=0)
                    if attn_sum_list[seg_idx] is None:
                        attn_sum_list[seg_idx] = avg_attn * X_batch.size(0)
                    else:
                        attn_sum_list[seg_idx] += avg_attn * X_batch.size(0)
                total_count += X_batch.size(0)
        all_logits = torch.cat(all_logits).numpy()
        all_features = torch.cat(all_features).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_eids = np.array(all_eids)
        result_df = pd.DataFrame({
            'eid': all_eids,
            'probability': all_logits,
            'label': all_targets
        })
        result_csv = f"risk_probabilities_{disease_name}.csv"
        result_df.to_csv(result_csv, index=False)
        logging.info(f"Prediction results saved to {result_csv}")
        binary_preds = (all_logits >= 0.5).astype(int)
        final_metrics = {
            'AUC': roc_auc_score(all_targets, all_logits),
            'Accuracy': accuracy_score(all_targets, binary_preds),
            'Precision': precision_score(all_targets, binary_preds, zero_division=0),
            'Recall': recall_score(all_targets, binary_preds, zero_division=0),
            'F1': f1_score(all_targets, binary_preds, zero_division=0)
        }
        cm = confusion_matrix(all_targets, binary_preds)
        final_metrics['TN'] = cm[0, 0]
        final_metrics['FP'] = cm[0, 1]
        final_metrics['FN'] = cm[1, 0]
        final_metrics['TP'] = cm[1, 1]
        with open(f'final_metrics_{disease_name}.txt', 'w') as f:
            f.write(f"Final metrics on full dataset for {disease_name}:\n")
            for metric, value in final_metrics.items():
                f.write(f"{metric}: {value}\n")
        avg_attn_list = [attn_sum / total_count for attn_sum in attn_sum_list]
        protein_attn = []
        protein_names = []
        num_true = len(common_proteins)
        for seg_idx in range(num_segments):
            true_len = segment_size if seg_idx < num_segments - 1 else (num_true - seg_idx * segment_size)
            protein_attn.extend(avg_attn_list[seg_idx][:true_len].detach().cpu().numpy().tolist())
            protein_names.extend(common_proteins[seg_idx * segment_size: seg_idx * segment_size + true_len])
        attn_df = pd.DataFrame({
            'protein': protein_names,
            'avg_attention': protein_attn
        })
        attn_csv = f"protein_avg_attention_{disease_name}.csv"
        attn_df.to_csv(attn_csv, index=False)
        logging.info(f"Protein average attention saved to {attn_csv}")
        max_plot_samples = 5000
        all_targets_np = all_targets
        indices_cases = np.where(all_targets_np == 1)[0]
        indices_controls = np.where(all_targets_np == 0)[0]
        if len(indices_controls) > len(indices_cases):
            np.random.seed(42)
            indices_controls_sample = np.random.choice(indices_controls, size=len(indices_cases), replace=False)
        else:
            indices_controls_sample = indices_controls
        selected_indices = np.concatenate([indices_cases, indices_controls_sample])
        if len(selected_indices) > max_plot_samples:
            np.random.seed(42)
            selected_indices = np.random.choice(selected_indices, size=max_plot_samples, replace=False)
            logging.info(f"Randomly sampling {max_plot_samples} points for visualization")
        np.random.shuffle(selected_indices)
        features_cluster = all_features[selected_indices]
        targets_cluster = all_targets_np[selected_indices]
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features_cluster)
        sns.set(style="whitegrid", font_scale=1.2)
        plt.figure(figsize=(8, 6))
        custom_cmap = ListedColormap(["#a6cee3", "#fb9a99"])
        plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=targets_cluster, cmap=custom_cmap,
                    alpha=0.7, edgecolors='black', linewidths=0.5)
        legend_elements = [
            Patch(facecolor="#fb9a99", edgecolor='black', label=disease_name.replace('_', ' ').title()),
            Patch(facecolor="#a6cee3", edgecolor='black', label="Healthy")
        ]
        leg = plt.legend(handles=legend_elements, title="Group", frameon=True)
        leg.get_frame().set_edgecolor('black')
        plt.title("TSNE Clustering")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        tsne_plot_file = f"tsne_{disease_name}.pdf"
        plt.savefig(tsne_plot_file, bbox_inches='tight')
        plt.close()
        logging.info(f"TSNE plot saved to {tsne_plot_file}")
        umap_embedder = umap.UMAP(random_state=42)
        features_umap = umap_embedder.fit_transform(features_cluster)
        plt.figure(figsize=(8, 6))
        plt.scatter(features_umap[:, 0], features_umap[:, 1], c=targets_cluster, cmap=custom_cmap,
                    alpha=0.7, edgecolors='black', linewidths=0.5)
        leg = plt.legend(handles=legend_elements, title="Group", frameon=True)
        leg.get_frame().set_edgecolor('black')
        plt.title("UMAP Clustering")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        umap_plot_file = f"umap_{disease_name}.pdf"
        plt.savefig(umap_plot_file, bbox_inches='tight')
        plt.close()
        logging.info(f"UMAP plot saved to {umap_plot_file}")
    summary_df = pd.DataFrame(metrics_summary_list)
    summary_df.to_csv("best_metrics_summary.csv", index=False)
    logging.info("Best metrics summary saved to best_metrics_summary.csv")
    logging.info("\n" + "=" * 80)
    logging.info("TRAINING COMPLETED - SUMMARY")
    logging.info("=" * 80)
    for metrics in metrics_summary_list:
        logging.info(f"\nDisease: {metrics['Disease']}")
        logging.info(f"Best Validation AUC: {metrics['Val_AUC']:.4f}")
        logging.info(f"Best Validation F1: {metrics['Val_F1']:.4f}")

if __name__ == '__main__':
    main()