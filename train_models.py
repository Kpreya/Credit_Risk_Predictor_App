# train_models.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import catboost as cb
import optuna

# Load paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'models')
PLOT_PATH = os.path.join(BASE_PATH, 'plots')

os.makedirs(PLOT_PATH, exist_ok=True)

# Load preprocessed data
X_train = joblib.load(os.path.join(MODEL_PATH, 'X_train_smote.joblib'))
y_train = joblib.load(os.path.join(MODEL_PATH, 'y_train_smote.joblib'))
X_test = joblib.load(os.path.join(MODEL_PATH, 'X_test_processed.joblib'))
y_test = joblib.load(os.path.join(MODEL_PATH, 'y_test.joblib'))

# Objective for LightGBM hyperparameter tuning
def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[valid_idx]
        y_tr, y_val = y_train[train_idx], y_train[valid_idx]
        model = lgb.LGBMClassifier(**params, random_state=42)
        # Fit without early stopping to avoid compatibility issues
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    return np.mean(aucs)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)
best_lgb_params = study_lgb.best_params
lgb_model = lgb.LGBMClassifier(**best_lgb_params, random_state=42)

# Objective for CatBoost hyperparameter tuning
def objective_cb(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 3, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'loss_function': 'Logloss'
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[valid_idx]
        y_tr, y_val = y_train[train_idx], y_train[valid_idx]
        model = cb.CatBoostClassifier(**params, random_state=42, verbose=False)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    return np.mean(aucs)

study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(objective_cb, n_trials=30)
best_cb_params = study_cb.best_params
cb_model = cb.CatBoostClassifier(**best_cb_params, random_state=42, verbose=False)

# Train base models
rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
cb_model.fit(X_train, y_train)

# Evaluate base models
models = {
    'Random Forest': rf_model,
    'LightGBM': lgb_model,
    'CatBoost': cb_model
}
results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba)
    })

# Stacking ensemble
estimators = [
    ('rf', rf_model),
    ('lgbm', lgb_model),
    ('catboost', cb_model)
]
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)
stack_model.fit(X_train, y_train)

y_pred = stack_model.predict(X_test)
y_proba = stack_model.predict_proba(X_test)[:, 1]
results.append({
    'Model': 'Stacking Ensemble',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC AUC': roc_auc_score(y_test, y_proba)
})

# Save models and results
joblib.dump(rf_model, os.path.join(MODEL_PATH, 'random_forest.joblib'))
joblib.dump(lgb_model, os.path.join(MODEL_PATH, 'lightgbm.joblib'))
joblib.dump(cb_model, os.path.join(MODEL_PATH, 'catboost.joblib'))
joblib.dump(stack_model, os.path.join(MODEL_PATH, 'stacking_ensemble.joblib'))

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODEL_PATH, 'model_comparison.csv'), index=False)

print("Model training and evaluation complete. Comparison saved to models/model_comparison.csv.")
