import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the list of diseases
diseases = ['parkinson', 'copd', 'dementia', 'stroke', 'asthma', 'glaucoma', 'ischaemic_heart_disease',
            'hypertension', 'atrial_fibrillation', 'heart_failure', 'cerebral_infarction', 'gout', 'obesity',
            'Colorectal_cancer', 'Skin_cancer', 'Breast_cancer', 'Lung_cancer', 'Prostate_cancer', 'RA',
            'diabetes', 'death']

# Custom dataset
class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Read protein data
protein_data = pd.read_csv('proteomic_data.csv')

# Assign diseases to GPUs
device_list = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
gpu_disease_dict = {device_list[i % 4]: diseases[i::4] for i in range(4)}

# Set some hyperparameters
batch_size = 256
num_epochs = 10
learning_rate = 1e-4

# Process the diseases allocated for each GPU
for device, disease_list in gpu_disease_dict.items():
    for disease in disease_list:
        print(f"Using device: {device} to train disease: {disease}")

        # Set the device
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Read the label data for the corresponding disease
        label_filename = f'{disease}_labels.csv'
        labels = pd.read_csv(label_filename)

        # Merge data
        data = pd.merge(protein_data, labels, on='eid')
        data.dropna(inplace=True)
        X = data.drop(['eid', 'label'], axis=1)
        y = data['label']

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the dataset
        X_train_np, X_test_np, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # Convert to Dataset
        train_dataset = ProteinDataset(X_train_np, y_train)
        test_dataset = ProteinDataset(X_test_np, y_test)

        # Define DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Define the MLP model
        class MLP(nn.Module):
            def __init__(self, input_size):
                super(MLP, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.model(x)

        # Initialize the model, loss function, and optimizer
        input_size = X_train_np.shape[1]
        model = MLP(input_size).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train_auc_list = []
        val_auc_list = []

        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Calculate AUC on the training set
            model.eval()
            with torch.no_grad():
                y_train_pred = []
                y_train_true = []
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch).squeeze()
                    y_train_pred.extend(outputs.cpu().numpy())
                    y_train_true.extend(y_batch.numpy())
                try:
                    train_auc = roc_auc_score(y_train_true, y_train_pred)
                except:
                    train_auc = float('nan')
                train_auc_list.append(train_auc)
                # Calculate AUC on the validation set
                y_val_pred = []
                y_val_true = []
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch).squeeze()
                    y_val_pred.extend(outputs.cpu().numpy())
                    y_val_true.extend(y_batch.numpy())
                try:
                    val_auc = roc_auc_score(y_val_true, y_val_pred)
                except:
                    val_auc = float('nan')
                val_auc_list.append(val_auc)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            y_test_pred = []
            y_test_true = []
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch).squeeze()
                y_test_pred.extend(outputs.cpu().numpy())
                y_test_true.extend(y_batch.numpy())

        # Convert predictions to binary labels
        y_pred_label = [1 if i >= 0.5 else 0 for i in y_test_pred]

        # Calculate metrics
        try:
            auc = roc_auc_score(y_test_true, y_test_pred)
        except:
            auc = float('nan')
        accuracy = accuracy_score(y_test_true, y_pred_label)
        precision = precision_score(y_test_true, y_pred_label, zero_division=0)
        recall = recall_score(y_test_true, y_pred_label, zero_division=0)
        f1 = f1_score(y_test_true, y_pred_label, zero_division=0)

        print(f"Disease: {disease}")
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Save the model
        model_filename = f'best_mlp_model_{disease}.pth'
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")