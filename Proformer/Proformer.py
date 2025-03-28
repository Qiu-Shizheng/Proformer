import os
import math
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Set random seed to ensure reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
available_gpu_count = torch.cuda.device_count()
if available_gpu_count < 4:
    print("Warning: Fewer than 4 GPUs available. The code will attempt to use the available GPU(s).")
else:
    print("Training will be performed on GPU0, GPU2, and GPU3.")

# 1. Data loading and preprocessing
class DementiaDataset(Dataset):
    def __init__(self, features, labels, eids):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.eids = eids  # Add eid list

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.eids[idx]  # Return eid

def load_data(features_path, labels_path):
    # Read feature data
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    # Ensure 'eid' column is unique
    features_df = features_df.drop_duplicates(subset=['eid'])
    labels_df = labels_df.drop_duplicates(subset=['eid'])

    # Merge datasets
    df = pd.merge(features_df, labels_df, on='eid')
    df = df.dropna()

    # Separate features and labels
    X = df.drop(['eid', 'label'], axis=1)
    y = df['label'].values
    eids = df['eid'].values  # Get eid

    return X, y, eids

# File paths
features_path = 'proteomic_data.csv'
labels_path = 't2d_labels.csv'
protein_corr_path = 'protein_correlation_matrix.csv'

X_df, y, eids = load_data(features_path, labels_path)
print(f"Feature dimensions: {X_df.shape}, Label dimensions: {y.shape}")

# Load protein correlation matrix
def load_protein_correlation_matrix(path):
    corr_df = pd.read_csv(path, index_col=0)
    return corr_df

# Load bias matrix
bias_df = load_protein_correlation_matrix(protein_corr_path)

# Get feature names and protein names from bias matrix
feature_names = X_df.columns.tolist()
bias_feature_names = bias_df.index.tolist()

# Get the intersection of feature names and bias matrix protein names
common_features = list(set(feature_names) & set(bias_feature_names))
print(f"Total number of common features: {len(common_features)}")

# Keep only common features
X_df = X_df[common_features]
bias_df = bias_df.loc[common_features, common_features]

# Convert features DataFrame to NumPy array
X = X_df.values

# Convert bias matrix to NumPy array and replace NaN with 0
bias_matrix = bias_df.fillna(0).values

# Check that the number of features matches the bias matrix dimensions
assert bias_matrix.shape[0] == X.shape[1], "Bias matrix dimensions do not match the number of features. Please check."

print(f"Updated feature dimensions: {X.shape}, Bias matrix shape: {bias_matrix.shape}")

# Split training and validation sets
X_train, X_val, y_train, y_val, eids_train, eids_val = train_test_split(
    X, y, eids, test_size=0.2, random_state=42, stratify=y
)

# Segment the features
num_features = X_train.shape[1]
segment_size = 512  # Each segment contains 512 features
num_segments = int(np.ceil(num_features / segment_size))
print(f"Features are segmented into {num_segments} segments, each segment length is {segment_size}.")

# Update bias_matrix: segment and pad the last segment if necessary
bias_matrices = []
for i in range(num_segments):
    start_idx = i * segment_size
    end_idx = (i + 1) * segment_size
    if end_idx > num_features:
        # Pad the last segment if necessary
        pad_size = end_idx - num_features
        bias_seg = bias_matrix[start_idx:num_features, start_idx:num_features]
        bias_seg = np.pad(bias_seg, ((0, pad_size), (0, pad_size)), 'constant', constant_values=0)
    else:
        bias_seg = bias_matrix[start_idx:end_idx, start_idx:end_idx]
    bias_matrices.append(bias_seg)

# Pad features so that each segment has a consistent number of features
def pad_features(X, segment_size):
    num_samples, total_features = X.shape
    pad_size = segment_size * num_segments - total_features
    if pad_size > 0:
        X_padded = np.pad(X, ((0, 0), (0, pad_size)), 'constant', constant_values=0)
    else:
        X_padded = X
    return X_padded

X_train_padded = pad_features(X_train, segment_size)
X_val_padded = pad_features(X_val, segment_size)
X_padded = pad_features(X, segment_size)  # The complete dataset's features also need padding

train_dataset = DementiaDataset(X_train_padded, y_train, eids_train)
val_dataset = DementiaDataset(X_val_padded, y_val, eids_val)

print(f"Padded training set feature dimensions: {X_train_padded.shape}")
print(f"Padded validation set feature dimensions: {X_val_padded.shape}")

# 2. Model Definition
class SegmentTransformer(nn.Module):
    def __init__(self, segment_size, d_token, n_heads, n_layers, dim_forward, dropout, bias_matrix):
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical=segment_size, d_token=d_token)

        seq_len = segment_size + 1  # +1 for CLS token

        # Convert bias matrix to tensor
        bias_tensor = torch.tensor(bias_matrix, dtype=torch.float32)
        assert bias_tensor.shape == (segment_size, segment_size), "Bias matrix size error"

        # Expand bias matrix dimensions to accommodate [CLS] token
        bias_tensor = nn.functional.pad(bias_tensor, (1, 0, 1, 0), value=0)  # (seq_len, seq_len)

        # Create encoder layers
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=dim_forward,
                dropout=dropout,
                bias_matrix=bias_tensor
            ) for _ in range(n_layers)
        ])

    def forward(self, x):
        tokens = self.tokenizer(x)  # (batch_size, seq_len, d_token)
        tokens = tokens.permute(1, 0, 2)  # (seq_len, batch_size, d_token)

        for layer in self.layers:
            tokens = layer(tokens)  # (seq_len, batch_size, d_token)

        cls_token = tokens[0, :, :]  # (batch_size, d_token)
        return cls_token

class Tokenizer(nn.Module):
    def __init__(
            self,
            d_numerical: int,
            d_token: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))  # +1 for [CLS] token
        self.bias_param = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))  # +1 for [CLS] token

        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn_init.zeros_(self.bias_param)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # Add [CLS] token with value 1
        cls_tokens = torch.ones(x_num.size(0), 1, device=x_num.device)
        x_num = torch.cat([cls_tokens, x_num], dim=1)  # (batch_size, num_features + 1)

        x = self.weight.unsqueeze(0) * x_num.unsqueeze(-1)  # (batch_size, num_features + 1, d_token)
        x = x + self.bias_param.unsqueeze(0)  # Broadcast addition of bias

        return x  # Output shape: (batch_size, num_features + 1, d_token)

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, bias_matrix=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)
        self.register_buffer('bias_matrix', bias_matrix)
        self.attn_weights = None  # Save attention weights

    def forward(self, query, key, value, **kwargs):
        attn_output, attn_output_weights = self.mha(query, key, value, need_weights=True, **kwargs)
        # Save attention weights before bias adjustment
        self.attn_weights = attn_output_weights.detach().cpu()

        # Add bias matrix to attention scores
        if self.bias_matrix is not None:
            # Expand bias matrix
            bias = self.bias_matrix.unsqueeze(0)  # (1, seq_len, seq_len)
            bias = bias.to(attn_output_weights.device)
            attn_output_weights = attn_output_weights + bias
            # Re-normalize attention weights
            attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
            # Recalculate attention output
            attn_output = torch.bmm(attn_output_weights, value.transpose(0, 1))
            attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_output_weights

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
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        # src shape: (seq_len, batch_size, d_model)
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerModel(nn.Module):
    def __init__(self,
                 n_features: int,
                 segment_size: int,
                 num_segments: int,
                 d_token: int,
                 n_heads: int,
                 n_layers: int,
                 dim_forward: int,
                 dropout: float,
                 bias_matrices: list):
        super().__init__()
        self.segment_size = segment_size
        self.num_segments = num_segments

        # Create a SegmentTransformer for each segment
        self.segment_transformers = nn.ModuleList([
            SegmentTransformer(
                segment_size=segment_size,
                d_token=d_token,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_forward=dim_forward,
                dropout=dropout,
                bias_matrix=bias_matrices[i]
            ) for i in range(num_segments)
        ])

        # Concatenated representation dimension
        total_d_token = d_token * num_segments

        self.classifier = nn.Linear(total_d_token, 1)

    def forward(self, x_num):
        segment_outputs = []
        for i in range(self.num_segments):
            start_idx = i * self.segment_size
            end_idx = (i + 1) * self.segment_size
            segment_input = x_num[:, start_idx:end_idx]  # (batch_size, segment_size)
            segment_output = self.segment_transformers[i](segment_input)  # (batch_size, d_token)
            segment_outputs.append(segment_output)

        # Concatenate outputs from all segments
        concatenated_output = torch.cat(segment_outputs, dim=-1)  # (batch_size, total_d_token)
        out = self.classifier(concatenated_output).squeeze(1)  # (batch_size)
        return torch.sigmoid(out)

# 3. Training and Validation Functions
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_targets = []
    for X_batch, y_batch, _ in tqdm(dataloader, desc="Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(y_batch.detach().cpu())
    epoch_loss /= len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    # Convert predicted probabilities to binary labels with threshold 0.5
    binary_preds = (all_preds >= 0.5).float()
    epoch_auc = roc_auc_score(all_targets, all_preds)
    epoch_acc = accuracy_score(all_targets, binary_preds)
    return epoch_loss, epoch_auc, epoch_acc

def eval_epoch(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch, _ in tqdm(dataloader, desc="Validation", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            epoch_loss += loss.item() * X_batch.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
    epoch_loss /= len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    # Convert predicted probabilities to binary labels with threshold 0.5
    binary_preds = (all_preds >= 0.5).float()
    epoch_auc = roc_auc_score(all_targets, all_preds)
    epoch_acc = accuracy_score(all_targets, binary_preds)
    return epoch_loss, epoch_auc, epoch_acc

# 4. Hyperparameter Optimization
def objective(trial):

    n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
    possible_d_tokens = [128, 256, 512, 1024]
    d_token = trial.suggest_categorical('d_token', [dt for dt in possible_d_tokens if dt % n_heads == 0])
    n_layers = trial.suggest_int('n_layers', 1, 8)
    dim_forward = trial.suggest_int('dim_forward', 128, 2048)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = TransformerModel(
        n_features=X_train_padded.shape[1],
        segment_size=segment_size,
        num_segments=num_segments,
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_forward=dim_forward,
        dropout=dropout,
        bias_matrices=bias_matrices
    ).to(device)

    if available_gpu_count >= 4:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=1)
        print(f"Training with {len([0, 1, 2, 3])} GPUs: GPU10, GPU1, GPU2, and GPU3")
    else:
        print("Fewer than 4 GPUs available, using a single GPU for training.")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training epochs
    epochs = 5
    best_auc = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_auc, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_auc, val_acc = eval_epoch(model, val_loader, criterion)
        print(f"Training loss: {train_loss:.4f}, Training AUC: {train_auc:.4f}, Training accuracy: {train_acc:.4f}")
        print(f"Validation loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}, Validation accuracy: {val_acc:.4f}")

        # Record best AUC
        if val_auc > best_auc:
            best_auc = val_auc
            # Save best model
            torch.save(model.state_dict(), 'best_model_tmp_t2d.pth')

        trial.report(val_auc, epoch)

        # Early stop if intermediate result is not promising
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_auc

# 5. Run Hyperparameter Optimization
import optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, timeout=36000)

print("Best parameters: ", study.best_params)
print("Best validation AUC: ", study.best_value)

# 6. Train Final Model Using Best Parameters
best_params = study.best_params
n_heads = best_params['n_heads']
d_token = best_params['d_token']
n_layers = best_params['n_layers']
dim_forward = best_params['dim_forward']
dropout = best_params['dropout']
lr = best_params['lr']
batch_size = best_params['batch_size']

# Create data loader (including all data)
full_dataset = DementiaDataset(X_padded, y, eids)
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Initialize model
model = TransformerModel(
    n_features=X_train_padded.shape[1],
    segment_size=segment_size,
    num_segments=num_segments,
    d_token=d_token,
    n_heads=n_heads,
    n_layers=n_layers,
    dim_forward=dim_forward,
    dropout=dropout,
    bias_matrices=bias_matrices
).to(device)

if available_gpu_count >= 4:
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=1)
    print(f"Training with {len([0, 1, 2, 3])} GPUs: GPU0, GPU1, GPU2, and GPU3")
else:
    print("Fewer than 4 GPUs available, using a single GPU for training.")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training epochs
epochs = 10
best_auc = 0
train_metrics_history = []
val_metrics_history = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss, train_auc, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_auc, val_acc = eval_epoch(model, val_loader, criterion)
    print(f"Training loss: {train_loss:.4f}, Training AUC: {train_auc:.4f}, Training accuracy: {train_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}, Validation accuracy: {val_acc:.4f}")

    train_metrics_history.append((train_auc, train_acc))
    val_metrics_history.append((val_auc, val_acc))

    # Save the best model
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model, 'best_final_model_t2d.pth')
        print("The entire model has been saved as 'best_final_model_t2d.pth'.")

# 7. Plot ROC Curve and Report Additional Metrics
# Load the saved model
model = torch.load('best_final_model_t2d.pth', map_location=device)
model.eval()

all_preds = []
all_targets = []
all_eids = []
with torch.no_grad():
    for X_batch, y_batch, eid_batch in tqdm(full_loader, desc="Computing predicted probabilities for all samples"):
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        all_preds.append(preds.cpu())
        all_targets.append(y_batch.cpu())
        all_eids.extend(eid_batch)

all_preds = torch.cat(all_preds).numpy()
all_targets = np.concatenate(all_targets)
all_eids = np.array(all_eids)

# Save predicted probabilities for all samples
result_df = pd.DataFrame({
    'eid': all_eids,
    'probability': all_preds,
    'label': all_targets
})
result_df.to_csv('risk_probabilities_t2d.csv', index=False)
print("Predicted probabilities for all samples have been saved to 'risk_probabilities_t2d.csv'.")

# Compute metrics on the validation set
val_preds = result_df[result_df['eid'].isin(eids_val)]
val_probs = val_preds['probability'].values
val_labels = val_preds['label'].values
val_binary_preds = (val_probs >= 0.5).astype(int)

# Calculate metrics
val_auc = roc_auc_score(val_labels, val_probs)
val_accuracy = accuracy_score(val_labels, val_binary_preds)
val_precision = precision_score(val_labels, val_binary_preds)
val_recall = recall_score(val_labels, val_binary_preds)
val_f1 = f1_score(val_labels, val_binary_preds)

print(f"Final validation metrics:")
print(f"AUC: {val_auc:.4f}")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1 Score: {val_f1:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label=f'AUC = {val_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC_T2D')
plt.legend(loc="lower right")
plt.savefig('roc_curve_t2d.pdf')
plt.show()

# 8. Model Saving
# The final model has already been saved during training as 'best_final_model_t2d.pth'
print("Model has been saved as 'best_final_model_t2d.pth'")