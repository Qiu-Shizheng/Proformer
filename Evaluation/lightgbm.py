import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

diseases = ['parkinson', 'copd', 'dementia', 'stroke', 'asthma',
            'glaucoma', 'ischaemic_heart_disease', 'hypertension', 'atrial_fibrillation',
            'heart_failure', 'cerebral_infarction', 'gout', 'obesity',
            'Colorectal_cancer', 'Skin_cancer', 'Breast_cancer', 'Lung_cancer',
            'Prostate_cancer', 'RA', 'diabetes', 'death']

# Read protein expression data
protein_data = pd.read_csv('proteomic_data.csv')

# Iterate over each disease
for disease in diseases:
    print(f"Processing disease: {disease}")

    # 1. Data reading
    # Read the label data for the corresponding disease
    labels = pd.read_csv(f'{disease}_labels.csv')

    # 2. Data merging
    # Merge features and labels by 'eid'
    data = pd.merge(protein_data, labels, on='eid')

    # 3. Feature and label extraction
    X = data.drop(['eid', 'label'], axis=1)
    y = data['label']

    # 4. Dataset splitting
    # Split the data into training and testing sets in an 8:2 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Create LightGBM dataset
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 6. Define initial parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1
    }
    # 7. Hyperparameter optimization
    # Define the parameter space
    param_dist = {
        'num_leaves': np.arange(20, 120, step=10),
        'max_depth': np.arange(3, 10, step=2),
        'learning_rate': np.linspace(0.01, 0.3, num=30),
        'n_estimators': np.arange(50, 400, step=50),
        'subsample': np.linspace(0.5, 1.0, num=6),
        'colsample_bytree': np.linspace(0.5, 1.0, num=6),
        'reg_alpha': np.linspace(0, 1.0, num=11),
        'reg_lambda': np.linspace(0, 1.0, num=11)
    }

    # Use RandomizedSearchCV for hyperparameter optimization
    clf = lgb.LGBMClassifier(**params)

    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # Start hyperparameter search
    random_search.fit(X_train, y_train)

    # Output best parameters
    print("Best parameters:", random_search.best_params_)
    print("Best AUC score:", random_search.best_score_)

    # Train the model using the best parameters
    best_params = random_search.best_params_
    model = lgb.LGBMClassifier(**params, **best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # 8. Model prediction
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # 9. Model evaluation
    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"Test set AUC: {auc_score}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 11. Save the model
    joblib.dump(model, f'lightgbm_{disease}_model.pkl')

    # 12. Save prediction results
    result = pd.DataFrame({
        'eid': data.loc[X_test.index, 'eid'],
        'label': y_test,
        'prediction': y_pred,
        'probability': y_pred_proba
    })
    result.to_csv(f'lightgbm_{disease}_prediction_results.csv', index=False)

    print(f"Model training and result saving for {disease} completed.\n")