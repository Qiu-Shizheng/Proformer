# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process

# 定义疾病列表
diseases = ['parkinson', 'copd', 'dementia', 'stroke', 'asthma', 'glaucoma', 'ischaemic_heart_disease', 'hypertension', 'atrial_fibrillation', 'heart_failure', 'cerebral_infarction', 'gout', 'obesity',  'Colorectal_cancer','Skin_cancer', 'Breast_cancer', 'Lung_cancer', 'Prostate_cancer', 'RA', 'diabetes', 'death']

# 定义可用的GPU设备ID列表
device_ids = [0, 1, 2, 3]

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        for layer in self.hidden_layers:
            residual = out
            out = layer(out)
            out = self.batch_norm(out)
            out = self.relu(out)
            out = self.dropout(out)
            out += residual  # 残差连接
        out = self.output_layer(out)
        return out

# 定义训练和评估函数
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    preds = []
    true_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds.extend(outputs.squeeze().cpu().numpy())
            true_labels.extend(y_batch.cpu().numpy())
    epoch_loss = running_loss / len(val_loader.dataset)
    auc = roc_auc_score(true_labels, preds)
    return epoch_loss, auc, preds, true_labels

# 定义超参数搜索范围
hyperparams = {
    'hidden_dim': [64, 128, 256],
    'num_layers': [1, 2, 3, 4],
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'batch_size': [512],
    'epochs': [10]
}

# 定义训练疾病的函数
def train_disease(disease, device_id):
    # 设置设备
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备：{device}，正在处理疾病：{disease}')

    # 读取数据
    labels = pd.read_csv(f'{disease}_labels.csv')
    data = pd.read_csv('filtered_20present_imputed_data.csv')

    # 合并数据集
    df = pd.merge(labels, data, on='eid')

    # 分离特征和标签
    X = df.drop(['eid', 'label'], axis=1)
    y = df['label']

    # 将数据转换为numpy数组
    X = X.values
    y = y.values

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 将数据转换为Tensor
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 将数据移动到设备
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    # 超参数优化
    from itertools import product
    import numpy as np

    best_auc = 0.0
    best_params = None
    best_model_state = None

    input_dim = X_train.shape[1]

    # K折交叉验证
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for hidden_dim, num_layers, dropout, lr, batch_size, epochs in product(
            hyperparams['hidden_dim'],
            hyperparams['num_layers'],
            hyperparams['dropout'],
            hyperparams['learning_rate'],
            hyperparams['batch_size'],
            hyperparams['epochs']
    ):
        fold_aucs = []
        print(f'\n疾病：{disease}，正在尝试参数组合：hidden_dim={hidden_dim}, num_layers={num_layers}, '
              f'dropout={dropout}, lr={lr}, batch_size={batch_size}, epochs={epochs}')

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train.cpu(), y_train.cpu()), 1):
            # 创建数据加载器
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            train_dataset = torch.utils.data.TensorDataset(X_fold_train, y_fold_train)
            val_dataset = torch.utils.data.TensorDataset(X_fold_val, y_fold_val)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

            # 实例化模型
            model = ResNet(input_dim, hidden_dim, num_layers, dropout).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # 学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=0.1, patience=5, verbose=True)

            # 训练模型
            best_fold_auc = 0.0
            for epoch in range(epochs):
                train_loss = train_model(model, train_loader, criterion, optimizer)
                val_loss, val_auc, _, _ = evaluate_model(model, val_loader, criterion)
                scheduler.step(val_loss)
                print(f'疾病：{disease}, Fold {fold}, Epoch {epoch + 1}/{epochs}, '
                      f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
                if val_auc > best_fold_auc:
                    best_fold_auc = val_auc
                    # 保存当前折的最佳模型参数
                    fold_best_model_state = model.state_dict()
            fold_aucs.append(best_fold_auc)
        avg_auc = np.mean(fold_aucs)
        print(f'疾病：{disease}，参数组合平均AUC: {avg_auc:.4f}')
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_params = {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': lr,
                'batch_size': batch_size,
                'epochs': epochs
            }
            best_model_state = fold_best_model_state  # 保存最佳模型参数

    print(f'\n疾病：{disease}，最佳参数组合: {best_params}')
    print(f'疾病：{disease}，最佳验证集平均AUC: {best_auc:.4f}')

    # 使用最佳参数在整个训练集上训练模型
    best_hidden_dim = best_params['hidden_dim']
    best_num_layers = best_params['num_layers']
    best_dropout = best_params['dropout']
    best_lr = best_params['learning_rate']
    best_batch_size = best_params['batch_size']
    best_epochs = best_params['epochs']

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=best_batch_size)

    model = ResNet(input_dim, best_hidden_dim, best_num_layers, best_dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.1, patience=5, verbose=True)

    # 加载最佳模型参数
    model.load_state_dict(best_model_state)

    # 训练模型并评估
    train_losses = []
    val_losses = []
    val_aucs = []

    for epoch in range(best_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_auc, test_preds, test_labels = evaluate_model(model, test_loader, criterion)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        print(f'疾病：{disease}, Epoch {epoch + 1}/{best_epochs}, '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

    # 打印分类报告和混淆矩阵
    test_preds_class = (np.array(test_preds) >= 0.5).astype(int)
    print(f"\n疾病：{disease}，测试集分类报告:")
    print(classification_report(test_labels, test_preds_class))
    print(f"疾病：{disease}，测试集混淆矩阵:")
    print(confusion_matrix(test_labels, test_preds_class))

    # 保存模型
    torch.save(model.state_dict(), f'best_resnet_model_{disease}.pth')
    print(f'模型已保存为 best_resnet_model_{disease}.pth')

# 按照GPU数量将疾病分组
disease_groups = [diseases[i:i+len(device_ids)] for i in range(0, len(diseases), len(device_ids))]

# 逐组处理疾病，每组中的疾病并行运行
for group in disease_groups:
    processes = []
    for i, disease in enumerate(group):
        device_id = device_ids[i % len(device_ids)]
        p = Process(target=train_disease, args=(disease, device_id))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()