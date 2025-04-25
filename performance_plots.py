# performance_plots.py
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay

# Paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'models')
PLOT_PATH  = os.path.join(BASE_PATH, 'plots')
os.makedirs(PLOT_PATH, exist_ok=True)

# Load test data + model
X_test = joblib.load(os.path.join(MODEL_PATH, 'X_test_processed.joblib'))
y_test = joblib.load(os.path.join(MODEL_PATH, 'y_test.joblib'))          # <– make sure you have y_test!
model  = joblib.load(os.path.join(MODEL_PATH, 'stacking_ensemble.joblib'))

# 1) Confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_PATH, 'stacking_ensemble_cm.png'), dpi=300)
plt.close(fig)

# 2) ROC curve
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
fig, ax = plt.subplots(figsize=(6,6))
roc_disp.plot(ax=ax)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_PATH, 'stacking_ensemble_roc.png'), dpi=300)
plt.close(fig)

print("Performance plots generated.")
