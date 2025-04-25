# shap_analysis.py (Using KernelExplainer for stacking ensemble)

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'models')
PLOT_PATH = os.path.join(BASE_PATH, 'plots')

os.makedirs(PLOT_PATH, exist_ok=True)

# Load test data and feature names
X_test_np = joblib.load(os.path.join(MODEL_PATH, 'X_test_processed.joblib'))
feature_names = joblib.load(os.path.join(MODEL_PATH, 'feature_names.joblib'))

# Convert to DataFrame
X_test = pd.DataFrame(X_test_np, columns=feature_names)

# Load stacking ensemble model
model = joblib.load(os.path.join(MODEL_PATH, 'stacking_ensemble.joblib'))

# Prepare background dataset for SHAP (subsample for efficiency)
background = shap.sample(X_test, 50, random_state=42)

# Define prediction function for positive class probability
def predict_proba_positive(data_matrix):
    # data_matrix can be numpy or pandas
    df = pd.DataFrame(data_matrix, columns=feature_names)
    return model.predict_proba(df)[:, 1]

# Initialize KernelExplainer with background data
explainer = shap.KernelExplainer(predict_proba_positive, background)

# Sample subset for SHAP computation
X_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)

# Compute SHAP values (nsamples controls runtime vs accuracy)
shap_values = explainer.shap_values(X_sample, nsamples=100)

# SHAP summary bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'stacking_ensemble_shap_summary_bar.png'), dpi=300)
plt.close()

# SHAP summary beeswarm plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_PATH, 'stacking_ensemble_shap_summary.png'), dpi=300)
plt.close()

print('SHAP analysis completed')
