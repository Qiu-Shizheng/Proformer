import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from catboost import CatBoostClassifier
import multiprocessing
from itertools import product
import joblib

def load_data(protein_path, label_path):
    """
    Load protein expression data and label data, and merge them together.
    """
    # Read protein expression data
    df_protein = pd.read_csv(protein_path)
    print(f"Protein expression data loaded, number of records: {df_protein.shape[0]}, number of features: {df_protein.shape[1]}")

    # Read label data
    df_label = pd.read_csv(label_path)
    print(f"Label data loaded, number of records: {df_label.shape[0]}")

    # Merge the datasets
    df = pd.merge(df_protein, df_label, on='eid')
    print(f"Merged dataset, number of records: {df.shape[0]}, number of features: {df.shape[1]}")

    return df

def preprocess_data(df):
    """
    Preprocess the data by separating features and labels.
    """
    # Separate features and labels
    X = df.drop(['eid', 'label'], axis=1)
    y = df['label']
    print(f"Feature matrix shape: {X.shape}, Label vector length: {len(y)}")
    return X, y

def get_hyperparameter_grid():
    """
    Define the hyperparameter grid.
    """
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [50, 100, 150],
        'l2_leaf_reg': [1, 3, 5]
    }
    # Generate all hyperparameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    return param_combinations

def evaluate_model(y_true, y_pred_proba, y_pred):
    """
    Calculate and return various evaluation metrics.
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return auc, accuracy, precision, recall, f1, report

def train_and_evaluate(params, X_train, y_train, X_val, y_val, gpu_id):
    """
    Train a CatBoost model and evaluate its performance.
    """
    # Set the environment variable to specify the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Define the model
    model = CatBoostClassifier(
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        iterations=params['iterations'],
        l2_leaf_reg=params['l2_leaf_reg'],
        eval_metric='AUC',
        random_seed=42,
        task_type='GPU',
        devices='0',
        verbose=10,
        early_stopping_rounds=10
    )

    # Train the model
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    # Predict probabilities and labels
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    # Calculate evaluation metrics
    auc, accuracy, precision, recall, f1, report = evaluate_model(y_val, y_pred_proba, y_pred)

    print(f"GPU{gpu_id} - Parameters: {params} => AUC: {auc:.4f}")

    return {
        'params': params,
        'model': model,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }

def worker(param_list, X_train, y_train, X_val, y_val, gpu_id, return_dict, process_id):
    best_result = {
        'auc': 0,
        'params': None,
        'model': None,
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'report': ''
    }
    for params in param_list:
        try:
            result = train_and_evaluate(params, X_train, y_train, X_val, y_val, gpu_id)
            if result['auc'] > best_result['auc']:
                best_result = result
        except Exception as e:
            print(f"GPU{gpu_id} - Parameters: {params} Training failed, error: {e}")

    # Store the best result in the return dictionary
    return_dict[process_id] = best_result

def hyperparameter_optimization_parallel(X_train, y_train, X_val, y_val, n_gpus=4):
    """
    Perform hyperparameter optimization using multi-GPU parallelism.
    """
    param_combinations = get_hyperparameter_grid()
    print(f"There are {len(param_combinations)} hyperparameter combinations to try.")

    # Allocate hyperparameter combinations to each GPU
    gpu_ids = list(range(n_gpus))
    param_split = [[] for _ in range(n_gpus)]
    for i, param in enumerate(param_combinations):
        param_split[i % n_gpus].append(param)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    for i in range(n_gpus):
        p = multiprocessing.Process(
            target=worker,
            args=(
                param_split[i],
                X_train,
                y_train,
                X_val,
                y_val,
                gpu_ids[i],
                return_dict,
                i
            )
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Collect the best results from all processes
    best_overall = {
        'auc': 0,
        'params': None,
        'model': None,
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'report': ''
    }

    for i in range(n_gpus):
        result = return_dict.get(i, {})
        if result and result['auc'] > best_overall['auc']:
            best_overall = result

    print(f"\nBest parameters: {best_overall['params']}, AUC: {best_overall['auc']:.4f}")
    return best_overall['model']

def train_model(X, y):
    """
    Split the training and validation sets, perform hyperparameter optimization, and train the model.
    """
    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")

    # Hyperparameter optimization
    best_model = hyperparameter_optimization_parallel(X_train, y_train, X_val, y_val, n_gpus=4)

    return best_model, X_val, y_val

def evaluate_model_final(model, X_val, y_val):
    """
    Evaluate the model performance and report the AUC and other common metrics.
    """
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    # Predict labels
    y_pred = model.predict(X_val)

    # Calculate metrics
    auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print("\nModel Evaluation Metrics:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))

def save_model_final(model, model_path, feature_importance_path):
    """
    Save the trained model and feature importance.
    """
    # Save the model
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")

    # Save the feature importance
    feature_importances = model.get_feature_importance()
    np.save(feature_importance_path, feature_importances)
    print(f"Feature importance saved to: {feature_importance_path}")

def main():
    # List of diseases
    diseases = [
        'parkinson', 'copd', 'dementia', 'stroke', 'asthma',
        'glaucoma', 'ischaemic_heart_disease', 'hypertension', 'atrial_fibrillation',
        'heart_failure', 'cerebral_infarction', 'gout', 'obesity',
        'Colorectal_cancer', 'Skin_cancer', 'Breast_cancer', 'Lung_cancer',
        'Prostate_cancer', 'RA', 'diabetes', 'death'
    ]

    # Protein expression file path
    protein_file = 'proteomic_data.csv'

    for disease in diseases:
        print(f"\n=== Now starting to process disease: {disease} ===")

        # Dynamically set the label file and model save paths
        label_file = f'{disease}_labels.csv'
        model_save_path = f'catboost_{disease}_model.cbm'
        feature_importance_save_path = f'feature_importances_{disease}.npy'

        # Check if the label file exists
        if not os.path.exists(label_file):
            print(f"Label file {label_file} does not exist, skipping disease: {disease}")
            continue

        # Load and merge the data
        df = load_data(protein_file, label_file)

        # Preprocess the data
        X, y = preprocess_data(df)

        # Check if both classes exist
        if y.nunique() != 2:
            print(f"The number of label classes for disease {disease} is not 2, skipping this disease.")
            continue

        # Train the model
        best_model, X_val, y_val = train_model(X, y)

        # Evaluate the model
        evaluate_model_final(best_model, X_val, y_val)

        # Save the model and feature importance
        save_model_final(best_model, model_save_path, feature_importance_save_path)

        print(f"=== Finished processing disease {disease} ===\n")

if __name__ == "__main__":
    main()