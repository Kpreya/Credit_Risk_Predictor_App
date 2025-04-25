# cnn_models.py

# This script trains a CNN feature extractor and hybrid models on preprocessed data.

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'models')
PLOT_PATH = os.path.join(BASE_PATH, 'plots')

os.makedirs(PLOT_PATH, exist_ok=True)

# Load preprocessed arrays
X_train_ps = joblib.load(os.path.join(MODEL_PATH, 'X_train_smote.joblib'))
X_test_ps = joblib.load(os.path.join(MODEL_PATH, 'X_test_processed.joblib'))
y_train = joblib.load(os.path.join(MODEL_PATH, 'y_train_smote.joblib'))
y_test = joblib.load(os.path.join(MODEL_PATH, 'y_test.joblib'))

# Reshape for CNN: (samples, timesteps, channels)
X_train_cnn = X_train_ps.reshape(X_train_ps.shape[0], X_train_ps.shape[1], 1)
X_test_cnn = X_test_ps.reshape(X_test_ps.shape[0], X_test_ps.shape[1], 1)

# Define CNN feature extractor

def create_cnn_feature_extractor(input_shape):
    inp = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inp)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Model(inputs=inp, outputs=x, name='cnn_feature_extractor')

# Build and train full CNN
input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])
feature_extractor = create_cnn_feature_extractor(input_shape)
outputs = Dense(1, activation='sigmoid')(feature_extractor.output)
full_model = Model(inputs=feature_extractor.input, outputs=outputs)
full_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
full_model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Extract features using trained extractor
feature_model = feature_extractor
X_train_features = feature_model.predict(X_train_cnn)
X_test_features = feature_model.predict(X_test_cnn)

# Save CNN feature extractor
feature_model.save(os.path.join(MODEL_PATH, 'cnn_feature_extractor.h5'))

# Evaluate hybrid models

def evaluate_hybrid(model, name):
    model.fit(X_train_features, y_train)
    y_pred = model.predict(X_test_features)
    y_proba = model.predict_proba(X_test_features)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {auc:.4f}")

    # Save hybrid model
    joblib.dump(model, os.path.join(MODEL_PATH, f"{name.lower().replace(' ', '_')}_ml.joblib"))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(os.path.join(PLOT_PATH, f"{name.lower().replace(' ', '_')}_cm.png"), dpi=300)
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"ROC Curve - {name}")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(PLOT_PATH, f"{name.lower().replace(' ', '_')}_roc.png"), dpi=300)
    plt.close()

# Hybrid models dictionary
hybrids = {
    'CNN-RF': RandomForestClassifier(random_state=42),
    'CNN-GB': GradientBoostingClassifier(random_state=42),
    'CNN-LR': LogisticRegression(max_iter=1000, random_state=42)
}

# Run evaluation for each hybrid model
for name, model in hybrids.items():
    evaluate_hybrid(model, name)

print('CNN hybrid models complete')
