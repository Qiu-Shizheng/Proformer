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

# Set random seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
available_gpu_count = torch.cuda.device_count()

# 1. Data Loading and Preprocessing
class DementiaDataset(Dataset):
    def __init__(self, features, labels, eids):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.eids = eids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.eids[idx]  # Return eid

def load_data(features_path, labels_path):
    # Read feature data
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    # Ensure the 'eid' column is unique
    features_df = features_df.drop_duplicates(subset=['eid'])
    labels_df = labels_df.drop_duplicates(subset=['eid'])

    # Merge datasets
    df = pd.merge(features_df, labels_df, on='eid')
    df = df.dropna()

    # Separate features and labels
    X = df.drop(['eid', 'label'], axis=1)
    y = df['label'].values
    eids = df['eid'].values  # Get eids

    return X, y, eids

# File paths
features_path = 'metabolomics.csv'
labels_path = 't2d_labels.csv'

X_df, y, eids = load_data(features_path, labels_path)
print(f"Feature dimensions: {X_df.shape}, Label dimensions: {y.shape}")

# Convert feature DataFrame to NumPy array
X = X_df.values

print(f"Updated feature dimensions: {X.shape}")

# Split training and validation sets
X_train, X_val, y_train, y_val, eids_train, eids_val = train_test_split(
    X, y, eids, test_size=0.2, random_state=42, stratify=y
)

train_dataset = DementiaDataset(X_train, y_train, eids_train)
val_dataset = DementiaDataset(X_val, y_val, eids_val)

print(f"Training set feature dimensions: {X_train.shape}")
print(f"Validation set feature dimensions: {X_val.shape}")

# 2. Model Definition
class Tokenizer(nn.Module):
    def __init__(self, d_numerical: int, d_token: int) -> None:
        super().__init__()
        # +1 for [CLS] token
        self.weight = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))
        self.bias_param = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))

        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn_init.zeros_(self.bias_param)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # Add [CLS] token with value 1
        cls_tokens = torch.ones(x_num.size(0), 1, device=x_num.device)
        x_num = torch.cat([cls_tokens, x_num], dim=1)  # (batch_size, num_features + 1)

        x = self.weight.unsqueeze(0) * x_num.unsqueeze(-1)  # (batch_size, num_features + 1, d_token)
        x = x + self.bias_param.unsqueeze(0)  # Broadcast bias addition
        return x  # Output shape: (batch_size, num_features + 1, d_token)

class TransformerModel(nn.Module):
    def __init__(self,
                 n_features: int,
                 d_token: int,
                 n_heads: int,
                 n_layers: int,
                 dim_forward: int,
                 dropout: float):
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical=n_features, d_token=d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=dim_forward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_token, 1)

    def forward(self, x_num):
        tokens = self.tokenizer(x_num)  # (batch_size, seq_len, d_token)
        tokens = tokens.permute(1, 0, 2)  # (seq_len, batch_size, d_token)
        tokens = self.transformer_encoder(tokens)
        cls_token = tokens[0, :, :]  # (batch_size, d_token)
        out = self.classifier(cls_token).squeeze(1)  # (batch_size)
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
    # Convert prediction probabilities to binary labels with threshold 0.5
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
        for X_batch, y_batch, _ in tqdm(dataloader, desc="Validating", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            epoch_loss += loss.item() * X_batch.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
    epoch_loss /= len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    # Convert prediction probabilities to binary labels with threshold 0.5
    binary_preds = (all_preds >= 0.5).float()
    epoch_auc = roc_auc_score(all_targets, all_preds)
    epoch_acc = accuracy_score(all_targets, binary_preds)
    return epoch_loss, epoch_auc, epoch_acc

# 4. Hyperparameter Optimization
def objective(trial):
    n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
    possible_d_tokens = [128, 256, 512]
    d_token = trial.suggest_categorical('d_token', [dt for dt in possible_d_tokens if dt % n_heads == 0])
    n_layers = trial.suggest_int('n_layers', 1, 8)
    dim_forward = trial.suggest_int('dim_forward', 128, 1024)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1024, 2048])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = TransformerModel(
        n_features=X_train.shape[1],
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_forward=dim_forward,
        dropout=dropout
    ).to(device)

    if available_gpu_count >= 4:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=0)
        print(f"Using {len([0, 1, 2, 3])} GPUs for training: GPU0, GPU1, GPU2 and GPU3")
    else:
        print("Fewer than 4 GPUs available; using available GPU(s).")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training epochs
    epochs = 5
    best_auc = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_auc, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_auc, val_acc = eval_epoch(model, val_loader, criterion)
        print(f"Training Loss: {train_loss:.4f}, Training AUC: {train_auc:.4f}, Training Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Record best AUC
        if val_auc > best_auc:
            best_auc = val_auc
            # Save the best model parameters
            torch.save(model.state_dict(), 'best_model_tmp_metabolomicsTransformer.pth')

        trial.report(val_auc, epoch)

        # Prune trial if intermediate result is unsatisfactory
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_auc

# 5. Run Hyperparameter Optimization
import optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5, timeout=36000)

print("Best Parameters: ", study.best_params)
print("Best Validation AUC: ", study.best_value)

# 6. Train Final Model with Best Hyperparameters
best_params = study.best_params
n_heads = best_params['n_heads']
d_token = best_params['d_token']
n_layers = best_params['n_layers']
dim_forward = best_params['dim_forward']
dropout = best_params['dropout']
lr = best_params['lr']
batch_size = best_params['batch_size']

# Create data loaders for full dataset
full_dataset = DementiaDataset(X, y, eids)
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Initialize model with best parameters
model = TransformerModel(
    n_features=X_train.shape[1],
    d_token=d_token,
    n_heads=n_heads,
    n_layers=n_layers,
    dim_forward=dim_forward,
    dropout=dropout
).to(device)

if available_gpu_count >= 4:
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=0)
    print(f"Using {len([0, 1, 2, 3])} GPUs for training: GPU0, GPU1, GPU2 and GPU3")
else:
    print("Fewer than 4 GPUs available; using available GPU(s).")

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
    print(f"Training Loss: {train_loss:.4f}, Training AUC: {train_auc:.4f}, Training Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}, Validation Accuracy: {val_acc:.4f}")

    train_metrics_history.append((train_auc, train_acc))
    val_metrics_history.append((val_auc, val_acc))

    # Save the best model
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model, 'best_final_model_metabolomicsTransformer.pth')
        print("Full model saved as 'best_final_model_metabolomicsTransformer.pth'.")

# 7. Plot ROC Curve and Report Other Metrics
# Load the saved model
model = torch.load('best_final_model_metabolomicsTransformer.pth', map_location=device)
model.eval()

all_preds = []
all_targets = []
all_eids = []
with torch.no_grad():
    for X_batch, y_batch, eid_batch in tqdm(full_loader, desc="Computing probabilities for all samples"):
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        all_preds.append(preds.cpu())
        all_targets.append(y_batch.cpu())
        all_eids.extend(eid_batch)

all_preds = torch.cat(all_preds).numpy()
all_targets = np.concatenate(all_targets)
all_eids = np.array(all_eids)

# Save the prediction probabilities for all samples
result_df = pd.DataFrame({
    'eid': all_eids,
    'probability': all_preds,
    'label': all_targets
})
result_df.to_csv('risk_probabilities_metabolomicsTransformer.csv', index=False)
print("The risk probabilities for all samples have been saved to 'risk_probabilities_metabolomicsTransformer.csv'.")

# Calculate metrics on the validation set
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

print("Final validation set metrics:")
print(f"AUC: {val_auc:.4f}")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1 Score: {val_f1:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {val_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_metabolomicsTransformer')
plt.legend(loc="lower right")
plt.savefig('roc_curve_metabolomicsTransformer.pdf')
plt.show()

# 8. Model Saving
print("Model saved as 'best_final_model_metabolomicsTransformer.pth'")
print("Model saved as 'best_final_model_metabolomicsTransformer.pth'")