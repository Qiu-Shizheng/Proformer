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
import optuna


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
available_gpu_count = torch.cuda.device_count()

# 定义疾病列表
diseases = [
    'parkinson', 'copd', 'dementia', 'stroke', 'asthma',
            'glaucoma', 'ischaemic_heart_disease', 'hypertension', 'atrial_fibrillation',
            'heart_failure', 'cerebral_infarction', 'gout', 'obesity',
            'Colorectal_cancer', 'Skin_cancer', 'Breast_cancer', 'Lung_cancer',
            'Prostate_cancer', 'RA', 'diabetes', 'death'
]


# 1. 数据加载与预处理
class DiseaseDataset(Dataset):
    def __init__(self, features, labels, eids):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.eids = eids  # 新增 eid 列表

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.eids[idx]  # 返回 eid


def load_data(features_path, labels_path):
    # 读取特征数据
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    # 确保 'eid' 列是唯一的
    features_df = features_df.drop_duplicates(subset=['eid'])
    labels_df = labels_df.drop_duplicates(subset=['eid'])

    # 合并数据集
    df = pd.merge(features_df, labels_df, on='eid')
    df = df.dropna()

    # 分离特征和标签
    X = df.drop(['eid', 'label'], axis=1)
    y = df['label'].values
    eids = df['eid'].values  # 获取 eid

    return X, y, eids


# 2. 模型定义
class Tokenizer(nn.Module):
    def __init__(self, d_numerical: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))  # +1 for [CLS] token
        self.bias_param = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))  # +1 for [CLS] token

        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn_init.zeros_(self.bias_param)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # 添加 [CLS] token，值为 1
        cls_tokens = torch.ones(x_num.size(0), 1, device=x_num.device)
        x_num = torch.cat([cls_tokens, x_num], dim=1)  # (batch_size, num_features + 1)

        x = self.weight.unsqueeze(0) * x_num.unsqueeze(-1)  # (batch_size, num_features + 1, d_token)
        x = x + self.bias_param.unsqueeze(0)  # 广播添加偏置

        return x  # 输出形状: (batch_size, num_features + 1, d_token)


class SegmentTransformer(nn.Module):
    def __init__(self, segment_size, d_token, n_heads, n_layers, dim_forward, dropout):
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical=segment_size, d_token=d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=dim_forward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        tokens = self.tokenizer(x)  # (batch_size, seq_len, d_token)
        tokens = tokens.permute(1, 0, 2)  # (seq_len, batch_size, d_token)

        tokens = self.transformer_encoder(tokens)

        cls_token = tokens[0, :, :]  # (batch_size, d_token)
        return cls_token


class TransformerModel(nn.Module):
    def __init__(self,
                 n_features: int,
                 segment_size: int,
                 num_segments: int,
                 d_token: int,
                 n_heads: int,
                 n_layers: int,
                 dim_forward: int,
                 dropout: float):
        super().__init__()
        self.segment_size = segment_size
        self.num_segments = num_segments

        # 为每个段创建一个 SegmentTransformer
        self.segment_transformers = nn.ModuleList([
            SegmentTransformer(
                segment_size=segment_size,
                d_token=d_token,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_forward=dim_forward,
                dropout=dropout
            ) for _ in range(num_segments)
        ])

        # 合并后的表示维度
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

        # 将所有段的输出拼接
        concatenated_output = torch.cat(segment_outputs, dim=-1)  # (batch_size, total_d_token)
        out = self.classifier(concatenated_output).squeeze(1)  # (batch_size)
        return torch.sigmoid(out)


# 3. 训练与验证函数
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_targets = []
    for X_batch, y_batch, _ in tqdm(dataloader, desc="训练中", leave=False):
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
    # 将预测概率转换为二进制标签，阈值为 0.5
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
        for X_batch, y_batch, _ in tqdm(dataloader, desc="验证中", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            epoch_loss += loss.item() * X_batch.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
    epoch_loss /= len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    # 将预测概率转换为二进制标签，阈值为 0.5
    binary_preds = (all_preds >= 0.5).float()
    epoch_auc = roc_auc_score(all_targets, all_preds)
    epoch_acc = accuracy_score(all_targets, binary_preds)
    return epoch_loss, epoch_auc, epoch_acc


# 4. 超参数优化
def objective(trial, train_dataset, val_dataset, segment_size, num_segments):
    n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
    possible_d_tokens = [128, 256, 512]
    d_token = trial.suggest_categorical('d_token', [dt for dt in possible_d_tokens if dt % n_heads == 0])
    n_layers = trial.suggest_int('n_layers', 1, 8)
    dim_forward = trial.suggest_int('dim_forward', 128, 1024)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = TransformerModel(
        n_features=train_dataset.X.shape[1],
        segment_size=segment_size,
        num_segments=num_segments,
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_forward=dim_forward,
        dropout=dropout
    ).to(device)


    if available_gpu_count >= 4:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=0)
        print(f"使用 {len([0, 1, 2, 3])} 个 GPU 进行训练：GPU1、GPU2 和 GPU3")
    else:
        print("可用的 GPU 数量不足 3 个，使用单个 GPU 进行训练。")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练周期
    epochs = 5
    best_auc = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_auc, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_auc, val_acc = eval_epoch(model, val_loader, criterion)
        print(f"训练损失: {train_loss:.4f}, 训练 AUC: {train_auc:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证 AUC: {val_auc:.4f}, 验证准确率: {val_acc:.4f}")

        # 记录最佳 AUC
        if val_auc > best_auc:
            best_auc = val_auc
            # 保存最佳模型
            torch.save(model.state_dict(), f'best_model_tmp_{disease}_no_ppi.pth')

        trial.report(val_auc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_auc


# 定义主处理函数
def process_disease(disease, features_path='proteomic_data.csv'):
    print(f"\n========== 开始处理疾病: {disease} ==========")
    labels_path = f'{disease}_labels.csv'

    if not os.path.exists(labels_path):
        print(f"标签文件 {labels_path} 不存在，跳过 {disease}")
        return

    # 1. 数据加载与预处理
    X_df, y, eids = load_data(features_path, labels_path)
    print(f"特征维度: {X_df.shape}, 标签维度: {y.shape}")

    X = X_df.values
    print(f"更新后特征维度: {X.shape}")

    # 拆分训练集和验证集
    X_train, X_val, y_train, y_val, eids_train, eids_val = train_test_split(
        X, y, eids, test_size=0.2, random_state=42, stratify=y
    )

    # 对特征进行分段
    num_features = X_train.shape[1]
    segment_size = 512
    num_segments = int(np.ceil(num_features / segment_size))
    print(f"特征被分为 {num_segments} 个段，每个段长度为 {segment_size}。")

    # 对特征进行填充，使每个段的特征数量一致
    def pad_features(X, segment_size, num_segments):
        pad_size = segment_size * num_segments - X.shape[1]
        if pad_size > 0:
            X_padded = np.pad(X, ((0, 0), (0, pad_size)), 'constant', constant_values=0)
        else:
            X_padded = X
        return X_padded

    X_train_padded = pad_features(X_train, segment_size, num_segments)
    X_val_padded = pad_features(X_val, segment_size, num_segments)
    X_padded = pad_features(X, segment_size, num_segments)  # 完整数据集的特征也需要填充

    train_dataset = DiseaseDataset(X_train_padded, y_train, eids_train)
    val_dataset = DiseaseDataset(X_val_padded, y_val, eids_val)

    print(f"填充后训练集特征维度: {X_train_padded.shape}")
    print(f"填充后验证集特征维度: {X_val_padded.shape}")

    # 4. 超参数优化
    def trial_objective(trial):
        return objective(trial, train_dataset, val_dataset, segment_size, num_segments)

    study = optuna.create_study(direction='maximize')
    study.optimize(trial_objective, n_trials=5, timeout=36000)

    print("最佳参数: ", study.best_params)
    print("最佳验证 AUC: ", study.best_value)

    # 6. 使用最佳参数训练最终模型
    best_params = study.best_params
    n_heads = best_params['n_heads']
    d_token = best_params['d_token']
    n_layers = best_params['n_layers']
    dim_forward = best_params['dim_forward']
    dropout = best_params['dropout']
    lr = best_params['lr']
    batch_size = best_params['batch_size']

    # 创建数据加载器
    full_dataset = DiseaseDataset(X_padded, y, eids)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = TransformerModel(
        n_features=X_train_padded.shape[1],
        segment_size=segment_size,
        num_segments=num_segments,
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_forward=dim_forward,
        dropout=dropout
    ).to(device)

    # 使用 GPU1、GPU2 和 GPU3
    if available_gpu_count >= 4:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=0)
        print(f"使用 {len([0, 1, 2, 3])} 个 GPU 进行训练：GPU1、GPU2 和 GPU3")
    else:
        print("可用的 GPU 数量不足 3 个，使用单个 GPU 进行训练。")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练周期
    epochs = 10
    best_auc = 0
    train_metrics_history = []
    val_metrics_history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_auc, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_auc, val_acc = eval_epoch(model, val_loader, criterion)
        print(f"训练损失: {train_loss:.4f}, 训练 AUC: {train_auc:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证 AUC: {val_auc:.4f}, 验证准确率: {val_acc:.4f}")

        train_metrics_history.append((train_auc, train_acc))
        val_metrics_history.append((val_auc, val_acc))

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model, f'best_final_model_{disease}_no_ppi.pth')
            print(f"整个模型已保存为 'best_final_model_{disease}_no_ppi.pth'。")

    # 7. 绘制 ROC 曲线和报告其他指标
    # 加载保存的模型
    model = torch.load(f'best_final_model_{disease}_no_ppi.pth', map_location=device)
    model.eval()

    all_preds = []
    all_targets = []
    all_eids = []
    with torch.no_grad():
        for X_batch, y_batch, eid_batch in tqdm(full_loader, desc="计算所有样本的概率"):
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
            all_eids.extend(eid_batch)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = np.concatenate(all_targets)
    all_eids = np.array(all_eids)

    # 保存所有样本的预测概率
    result_df = pd.DataFrame({
        'eid': all_eids,
        'probability': all_preds,
        'label': all_targets
    })
    result_df.to_csv(f'risk_probabilities_{disease}_no_ppi.csv', index=False)
    print(f"所有样本的患病概率已保存到 'risk_probabilities_{disease}_no_ppi.csv'。")

    # 在验证集上计算指标
    val_preds = result_df[result_df['eid'].isin(eids_val)]
    val_probs = val_preds['probability'].values
    val_labels = val_preds['label'].values
    val_binary_preds = (val_probs >= 0.5).astype(int)

    # 计算指标
    val_auc = roc_auc_score(val_labels, val_probs)
    val_accuracy = accuracy_score(val_labels, val_binary_preds)
    val_precision = precision_score(val_labels, val_binary_preds)
    val_recall = recall_score(val_labels, val_binary_preds)
    val_f1 = f1_score(val_labels, val_binary_preds)

    print(f"\n最终验证集上的指标：")
    print(f"AUC: {val_auc:.4f}")
    print(f"准确率: {val_accuracy:.4f}")
    print(f"精确率: {val_precision:.4f}")
    print(f"召回率: {val_recall:.4f}")
    print(f"F1 得分: {val_f1:.4f}")

    # 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'AUC = {val_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC_{disease}_no_ppi')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{disease}_no_ppi.pdf')
    plt.show()

    print(f"模型已保存为 'best_final_model_{disease}_no_ppi.pth'\n")


# 5. 主循环，遍历所有疾病
for disease in diseases:
    process_disease(disease)