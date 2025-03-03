import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import optuna

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading
protein_data = pd.read_csv('proteomic_data.csv')

# Define list of diseases
diseases = ['parkinson', 'copd', 'dementia', 'stroke', 'asthma',
            'glaucoma', 'ischaemic_heart_disease', 'hypertension', 'atrial_fibrillation',
            'heart_failure', 'cerebral_infarction', 'gout', 'obesity',
            'Colorectal_cancer', 'Skin_cancer', 'Breast_cancer', 'Lung_cancer',
            'Prostate_cancer', 'RA', 'diabetes', 'death']

# Create dataset and dataloader class
class DementiaDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define the Tokenizer class
class Tokenizer(nn.Module):
    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        self.d_numerical = d_numerical
        self.weight = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))  # Shape: (n_tokens, d_token)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x_num, x_cat=None):
        # Add [CLS] token
        batch_size = x_num.size(0)
        x_num = torch.cat([torch.ones(batch_size, 1, device=x_num.device), x_num], dim=1)  # Shape: (batch_size, n_tokens)
        x = x_num.unsqueeze(-1) * self.weight.unsqueeze(0)  # Shape: (batch_size, n_tokens, d_token)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)  # Shape: (batch_size, n_tokens, d_token)
        return x

# Define the MultiheadAttention class
class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization):
        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        nn.init.xavier_uniform_(self.W_q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_k.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_v.weight, gain=1 / math.sqrt(2))
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def forward(self, x_q, x_kv):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        batch_size, n_tokens, d = q.shape
        d_head = d // self.n_heads

        q = q.view(batch_size, n_tokens, self.n_heads, d_head).transpose(1, 2)
        k = k.view(batch_size, n_tokens, self.n_heads, d_head).transpose(1, 2)
        v = v.view(batch_size, n_tokens, self.n_heads, d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
        attention = torch.softmax(scores, dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ v

        x = x.transpose(1, 2).contiguous().view(batch_size, n_tokens, d)
        if self.W_out is not None:
            x = self.W_out(x)
        return x

# Define the Transformer class
class Transformer(nn.Module):
    def __init__(self, *, d_numerical, categories, token_bias, n_layers, d_token, n_heads,
                 d_ffn_factor, attention_dropout, ffn_dropout, residual_dropout,
                 activation, prenormalization, initialization, kv_compression,
                 kv_compression_sharing, d_out):
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                'attention': MultiheadAttention(d_token, n_heads, attention_dropout, initialization),
                'linear0': nn.Linear(d_token, int(d_token * d_ffn_factor)),
                'linear1': nn.Linear(int(d_token * d_ffn_factor), d_token),
                'norm1': nn.LayerNorm(d_token),
                'norm0': nn.LayerNorm(d_token),
            })
            self.layers.append(layer)

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.prenormalization = prenormalization
        self.last_normalization = nn.LayerNorm(d_token) if prenormalization else None
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num, x_cat=None):
        x = self.tokenizer(x_num, x_cat)
        for layer in self.layers:
            x_residual = x
            x_residual = layer['norm0'](x_residual)
            x_residual = layer['attention'](x_residual, x_residual)
            x = x + x_residual
            x_residual = x
            x_residual = layer['norm1'](x_residual)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation_fn(x_residual)
            x_residual = layer['linear1'](x_residual)
            x = x + x_residual
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.head(x[:, 0, :])  # Take the output of the [CLS] token
        return x

# Define batch size
batch_size = 32

# Loop through each disease
for disease in diseases:
    print(f"\nProcessing disease: {disease}")

    # Load the corresponding disease labels file
    labels = pd.read_csv(f'{disease}_labels.csv')

    # Merge data based on the 'eid' field
    data = pd.merge(protein_data, labels, on='eid')

    # If there are missing values, skip this disease
    if data.isnull().values.any():
        print(f"Missing values exist, skipping disease: {disease}")
        continue

    # Extract features and labels
    X = data.drop(['eid', 'label'], axis=1).values  # Protein features
    y = data['label'].values  # Labels, 0 or 1

    # Split the dataset into training, validation, and test sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)

    # Data normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    # Create dataset and dataloader
    train_dataset = DementiaDataset(X_train_tensor, y_train_tensor)
    val_dataset = DementiaDataset(X_val_tensor, y_val_tensor)
    test_dataset = DementiaDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define the objective function for hyperparameter optimization
    def objective(trial):
        # Define hyperparameter search space
        n_layers = trial.suggest_int('n_layers', 1, 8)
        d_token = trial.suggest_categorical('d_token', [128, 256, 512])
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        d_ffn_factor = trial.suggest_uniform('d_ffn_factor', 2.0, 4.0)
        attention_dropout = trial.suggest_uniform('attention_dropout', 0.0, 0.5)
        ffn_dropout = trial.suggest_uniform('ffn_dropout', 0.0, 0.5)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        activation = trial.suggest_categorical('activation', ['relu', 'gelu'])

        # Initialize the model
        d_input = X_train.shape[1]
        n_classes = 1

        model = Transformer(
            d_numerical=d_input,
            categories=None,
            token_bias=True,
            n_layers=n_layers,
            d_token=d_token,
            n_heads=n_heads,
            d_ffn_factor=d_ffn_factor,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=0.0,
            activation=activation,
            prenormalization=True,
            initialization='kaiming',
            kv_compression=None,
            kv_compression_sharing=None,
            d_out=n_classes
        )

        # Move the model to the device
        model = model.to(device)

        # Use DataParallel for multi-GPU parallelism
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # Define the loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training and validation
        n_epochs = 5
        for epoch in range(n_epochs):
            model.train()
            train_losses = []
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_features, None)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_targets = []
            val_probs = []
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_features, None)
                    val_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(batch_labels.cpu().numpy())

            # If there is only one class, skip
            if len(np.unique(val_targets)) == 1:
                return 0.0

            val_auc = roc_auc_score(val_targets, val_probs)

            # Use validation AUC as the metric
            trial.report(val_auc, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_auc

    # Create study object and perform optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)

    print('Best hyperparameters:')
    print(study.best_params)
    print(f'Best validation AUC: {study.best_value:.4f}')

    # Retrain the model using the best hyperparameters
    best_params = study.best_params

    # Initialize the model
    d_input = X_train.shape[1]
    n_classes = 1

    model = Transformer(
        d_numerical=d_input,
        categories=None,
        token_bias=True,
        n_layers=best_params['n_layers'],
        d_token=best_params['d_token'],
        n_heads=best_params['n_heads'],
        d_ffn_factor=best_params['d_ffn_factor'],
        attention_dropout=best_params['attention_dropout'],
        ffn_dropout=best_params['ffn_dropout'],
        residual_dropout=0.0,
        activation=best_params['activation'],
        prenormalization=True,
        initialization='kaiming',
        kv_compression=None,
        kv_compression_sharing=None,
        d_out=n_classes
    )

    # Move the model to the device
    model = model.to(device)

    # Use DataParallel for multi-GPU parallelism
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

    # Training loop
    n_epochs = 10

    best_val_auc = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features, None)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_targets = []
        val_probs = []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features, None)
                val_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(batch_labels.cpu().numpy())

        # If there is only one class, skip
        if len(np.unique(val_targets)) == 1:
            print(f"Only one class in the validation set, skipping disease: {disease}")
            break

        val_auc = roc_auc_score(val_targets, val_probs)

        print(f"Epoch {epoch}/{n_epochs}, Training loss: {avg_train_loss:.4f}, Validation AUC: {val_auc:.4f}")

        # Save the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'best_model_ft{disease}.pt')
            print("Best model saved")

    # Test set evaluation
    model.load_state_dict(torch.load(f'best_model_ft{disease}.pt'))
    model.eval()
    test_targets = []
    test_probs = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features, None)
            test_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            test_targets.extend(batch_labels.cpu().numpy())

    # If there is only one class, skip
    if len(np.unique(test_targets)) == 1:
        print(f"Only one class in the test set, unable to compute metrics, skipping disease: {disease}")
        continue

    test_auc = roc_auc_score(test_targets, test_probs)
    test_accuracy = accuracy_score(test_targets, np.round(test_probs))
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, np.round(test_probs), average='binary')
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    print(f"Test F1 score: {f1:.4f}")

    # Save final model results
    torch.save(model.state_dict(), f'final_model_FT_Transformer{disease}.pt')
    print(f"Final model has been saved as 'final_model_FT_Transformer{disease}.pt'.")