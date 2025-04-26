# train_models.py

# Credit Risk Project - Model Training Script

# Imports 
import os
import warnings
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Config 
warnings.filterwarnings("ignore")

# Paths 
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_PATH, 'models')

#  Load Preprocessed Data 
try:
    X_train = joblib.load(os.path.join(MODEL_PATH, 'X_train_smote.joblib'))
    y_train = joblib.load(os.path.join(MODEL_PATH, 'y_train_smote.joblib'))
except FileNotFoundError as e:
    raise Exception("Required preprocessed files not found. Please run 'creditrisk.py' first to generate them.") from e

#  Subsampling 
# To make training faster on large data. Adjust n_samples if needed.
from sklearn.utils import resample
X_sampled, y_sampled = resample(X_train, y_train, n_samples=int(0.5 * len(X_train)), random_state=42)

#  Model Definitions 
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(max_iter=500, solver='liblinear', random_state=42),
    'SVC': SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
}

#  Model Training 
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_sampled, y_sampled)
    joblib.dump(model, os.path.join(MODEL_PATH, f"{name}_model.joblib"))

print("All models trained and saved successfully.")

# Cross-validation Scores 
print("\nCross-validation Results:")
for name, model in models.items():
    scores = cross_val_score(model, X_sampled, y_sampled, cv=5, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

