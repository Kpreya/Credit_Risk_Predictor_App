# creditrisk.py
# -*- coding: utf-8 -*-

# Credit Risk Project - Preprocessing Script

# ==================== Imports ====================
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# ==================== Config ====================
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

# ==================== Paths ====================
BASE_PATH       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(BASE_PATH, 'data', 'german_credit_data.csv')
PLOT_PATH       = os.path.join(BASE_PATH, 'plots')
MODEL_PATH      = os.path.join(BASE_PATH, 'models')
PREPROCESSED_PATH = os.path.join(BASE_PATH, 'data', 'preprocessed_credit_data.csv')

os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# ==================== Load Data ====================
df = pd.read_csv(DATA_PATH)

# ==================== Clean Columns ====================
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
if 'unnamed:_0' in df.columns:
    df.drop(columns=['unnamed:_0'], inplace=True)

# ==================== Impute & Clean ====================
cat_cols = ['sex', 'job', 'housing', 'saving_accounts', 'checking_account', 'purpose']
num_cols = ['age', 'credit_amount', 'duration']

cat_imp = SimpleImputer(strategy='most_frequent')
df[cat_cols] = pd.DataFrame(cat_imp.fit_transform(df[cat_cols]), columns=cat_cols)

num_imp = SimpleImputer(strategy='median')
df[num_cols] = pd.DataFrame(num_imp.fit_transform(df[num_cols]), columns=num_cols)

# ==================== Outlier Capping ====================
def cap_outliers(df, cols):
    df_capped = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_capped[col] = df_capped[col].clip(lower, upper)
    return df_capped

df = cap_outliers(df, num_cols)

# ==================== Feature Engineering ====================
df['credit_per_month']      = df['credit_amount'] / (df['duration'] + 1e-3)
df['age_group']             = pd.cut(df['age'],
                                      bins=[18, 30, 40, 50, 60, 100],
                                      labels=['18-30', '31-40', '41-50', '51-60', '60+'])
df['credit_to_age_ratio']   = df['credit_amount'] / df['age']
df['saving_accounts']       = df['saving_accounts'].fillna('none')
df['checking_account']      = df['checking_account'].fillna('none')

saving_map  = {'none': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4}
checking_map = {'none': 0, 'little': 1, 'moderate': 2, 'rich': 3}
df['saving_level']         = df['saving_accounts'].map(saving_map)
df['checking_level']       = df['checking_account'].map(checking_map)
df['financial_security']   = df['saving_level'] / (df['credit_amount'] / 1000 + 1)

cat_cols += ['age_group']
num_cols += ['credit_per_month', 'credit_to_age_ratio', 'financial_security']

# ==================== Save Preprocessed Data ====================
df.to_csv(PREPROCESSED_PATH, index=False)

# ==================== Encode Target & Split ====================
le = LabelEncoder().fit(df['risk'])
y  = le.transform(df['risk'])
joblib.dump(le, os.path.join(MODEL_PATH, 'target_encoder.joblib'))

X = df.drop(columns=['risk'])
cat_cols_final = [col for col in cat_cols if col in X.columns]
num_cols_final = [col for col in num_cols if col in X.columns]

# ==================== Preprocessing Pipelines ====================
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols_final),
    ('cat', cat_pipeline, cat_cols_final)
])

# ==================== Train/Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

# ==================== Handle Imbalance with SMOTE ====================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

# ==================== Persist Artifacts ====================
joblib.dump(preprocessor, os.path.join(MODEL_PATH, 'preprocessor.joblib'))

# Serialize arrays for downstream scripts
joblib.dump(X_train_smote,      os.path.join(MODEL_PATH, 'X_train_smote.joblib'))
joblib.dump(y_train_smote,      os.path.join(MODEL_PATH, 'y_train_smote.joblib'))
joblib.dump(X_test_processed,   os.path.join(MODEL_PATH, 'X_test_processed.joblib'))
joblib.dump(y_test,             os.path.join(MODEL_PATH, 'y_test.joblib'))

# Prepare CNN inputs
X_train_cnn = X_train_smote.reshape(X_train_smote.shape[0], 1, X_train_smote.shape[1])
X_test_cnn  = X_test_processed.reshape(X_test_processed.shape[0], 1, X_test_processed.shape[1])

joblib.dump(X_train_cnn, os.path.join(MODEL_PATH, 'X_train_cnn.joblib'))
joblib.dump(X_test_cnn,  os.path.join(MODEL_PATH, 'X_test_cnn.joblib'))

# Dump feature names for interpretation
ohe_cols      = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols_final)
feature_names = num_cols_final + list(ohe_cols)
joblib.dump(feature_names, os.path.join(MODEL_PATH, 'feature_names.joblib'))

print(" Preprocessing complete and all artifacts saved to models/.")
