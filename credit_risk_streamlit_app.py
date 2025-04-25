# credit_risk_streamlitapp.py
# Streamlit application for credit risk prediction and insights

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from PIL import Image

# ==================== Config ====================
st.set_page_config(
    page_title="Credit Risk Predictor",
    layout="wide"
)

# ==================== Paths ====================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data', 'preprocessed_credit_data.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'models')
PLOT_PATH = os.path.join(BASE_PATH, 'plots')

# ==================== Load Data & Models ====================
df = pd.read_csv(DATA_PATH)
preprocessor = joblib.load(os.path.join(MODEL_PATH, 'preprocessor.joblib'))
label_encoder = joblib.load(os.path.join(MODEL_PATH, 'target_encoder.joblib'))
model = joblib.load(os.path.join(MODEL_PATH, 'stacking_ensemble.joblib'))

# ==================== Sidebar - User Input ====================
st.sidebar.header("Input Features")
sex = st.sidebar.selectbox("Sex", df['sex'].unique())
age = st.sidebar.slider("Age", int(df['age'].min()), int(df['age'].max()), int(df['age'].median()))
job = st.sidebar.selectbox("Job", df['job'].unique())
housing = st.sidebar.selectbox("Housing", df['housing'].unique())
saving_accounts = st.sidebar.selectbox("Saving Accounts", df['saving_accounts'].unique())
checking_account = st.sidebar.selectbox("Checking Account", df['checking_account'].unique())
credit_amount = st.sidebar.number_input("Credit Amount", min_value=int(df['credit_amount'].min()), max_value=int(df['credit_amount'].max()), value=int(df['credit_amount'].median()))
duration = st.sidebar.slider("Duration (months)", int(df['duration'].min()), int(df['duration'].max()), int(df['duration'].median()))
purpose = st.sidebar.selectbox("Purpose", df['purpose'].unique())

input_dict = {
    'sex': sex,
    'age': age,
    'job': job,
    'housing': housing,
    'saving_accounts': saving_accounts,
    'checking_account': checking_account,
    'credit_amount': credit_amount,
    'duration': duration,
    'purpose': purpose
}
input_df = pd.DataFrame([input_dict])

# Feature engineering on input
input_df['credit_per_month'] = input_df['credit_amount'] / (input_df['duration'] + 1e-3)
input_df['age_group'] = pd.cut(
    input_df['age'], bins=[18, 30, 40, 50, 60, 100],
    labels=['18-30', '31-40', '41-50', '51-60', '60+']
)
input_df['credit_to_age_ratio'] = input_df['credit_amount'] / input_df['age']
input_df['saving_level'] = input_df['saving_accounts'].map({'none': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4})
input_df['checking_level'] = input_df['checking_account'].map({'none': 0, 'little': 1, 'moderate': 2, 'rich': 3})
input_df['financial_security'] = input_df['saving_level'] / (input_df['credit_amount'] / 1000 + 1)

# Predict
X_proc = preprocessor.transform(input_df)
pred = model.predict(X_proc)[0]
proba = model.predict_proba(X_proc)[0][1]
pred_label = label_encoder.inverse_transform([pred])[0]

# ==================== Main Page Layout ====================
tabs = st.tabs(["Prediction", "Exploration", "Performance", "Interpretation"])

# --- Prediction Tab ---
# new

probas = model.predict_proba(X_proc)[0]       # [P(bad), P(good)]
proba_bad = probas[0]
proba_good = probas[1]

# you can use a slider-threshold or fixed 0.5
threshold = st.sidebar.slider(
    "Bad-risk probability threshold",
    0.0, 1.0, 0.5, step=0.01
)
threshold = 0.5
pred_label = 'bad' if proba_bad >= threshold else 'good'

# display
if pred_label == 'bad':
    st.error("🔴 Predicted Risk: Bad Credit")
else:
    st.success("🟢 Predicted Risk: Good Credit")

# show the correct bad-risk probability
st.write(f"Probability of **bad** risk: {proba_bad:.2%}")


# --- Exploration Tab ---
with tabs[1]:
    st.header("Data Exploration")
    col = st.selectbox("Select feature to explore", ['age', 'credit_amount', 'duration', 'financial_security'])
    fig = px.histogram(
        df, x=col, color='risk', barmode='overlay', marginal='box',
        title=f'Distribution of {col} by Risk'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Performance Tab ---
with tabs[2]:
    st.header("Model Performance")
    # Load and display confusion matrix and ROC
    cm_img = Image.open(os.path.join(PLOT_PATH, 'stacking_ensemble_cm.png'))
    roc_img = Image.open(os.path.join(PLOT_PATH, 'stacking_ensemble_roc.png'))
    st.subheader("Confusion Matrix")
    st.image(cm_img)
    st.subheader("ROC Curve")
    st.image(roc_img)

# --- Interpretation Tab ---
with tabs[3]:
    st.header("Model Interpretation")
    st.write("### SHAP Summary Bar")
    sb_img = Image.open(os.path.join(PLOT_PATH, 'stacking_ensemble_shap_summary_bar.png'))
    st.image(sb_img, caption='SHAP Feature Importance (bar)')
    st.write("### SHAP Summary")
    s_img = Image.open(os.path.join(PLOT_PATH, 'stacking_ensemble_shap_summary.png'))
    st.image(s_img, caption='SHAP Summary Plot')
def load_plot(name, caption=None):
    path = os.path.join(PLOT_PATH, name)
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, caption=caption)
    else:
        st.warning(f"⚠️ Plot `{name}` not found.")

# … in your tabs[2] “Performance”:
with tabs[2]:
    st.header("Model Performance")
    st.subheader("Confusion Matrix")
    load_plot('stacking_ensemble_cm.png')
    st.subheader("ROC Curve")
    load_plot('stacking_ensemble_roc.png')


# ==================== Footer ====================
st.write("---")
st.write("Designed and implemented by Krishnopreya.")
