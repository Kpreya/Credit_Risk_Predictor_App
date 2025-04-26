# credit_risk_streamlitapp.py
# Streamlit application for Credit Risk Prediction and Insights

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from PIL import Image

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="Credit Risk Predictor | AI-Powered Credit Check",
    page_icon="🏦",
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

# ==================== Introduction / Overview ====================
st.title("🏦 AI-Powered Credit Risk Predictor")
st.markdown("""
#### Instantly Predict Customer Creditworthiness with AI  
Welcome to the **Credit Risk Predictor** – your smart tool to assess customer credit risk with **real-time predictions** and **data-driven insights**.  
Built with an advanced **Stacking Ensemble** model, achieving **86% accuracy**, this app empowers you to make **faster**, **smarter**, and **risk-free** credit decisions. 

---
""")

with st.expander("📖 What This App Does ", expanded=True):
    st.markdown("""
    -  **Predict** whether a customer is likely to be a **Good** or **Bad** credit risk
    -  **Model Used:** Stacking Ensemble (combines strengths of multiple models)
    -  **Accuracy:** ~86% on real-world test data
    -  **Key Insights:** 
        - Lower savings, higher loan amount, and longer loan durations increase bad risk chances.
        - Age, job type, and account balance are strong indicators.
    -  **How it Works:** Fill the customer details on the sidebar → Instantly get prediction + risk probability!
    
    **Built for:** Loan officers, banks, financial advisors, fintech apps, and anyone who wants fast credit decisions.
    """)

# ==================== Sidebar - User Input ====================
st.sidebar.header("📋 Enter Customer Details")

sex = st.sidebar.selectbox("Sex", df['sex'].unique())
age = st.sidebar.slider("Age", int(df['age'].min()), int(df['age'].max()), int(df['age'].median()))
job = st.sidebar.selectbox("Job", df['job'].unique())
housing = st.sidebar.selectbox("Housing Status", df['housing'].unique())
saving_accounts = st.sidebar.selectbox("Saving Accounts Status", df['saving_accounts'].unique())
checking_account = st.sidebar.selectbox("Checking Account Status", df['checking_account'].unique())
credit_amount = st.sidebar.number_input("Requested Credit Amount", min_value=int(df['credit_amount'].min()), max_value=int(df['credit_amount'].max()), value=int(df['credit_amount'].median()))
duration = st.sidebar.slider("Credit Duration (Months)", int(df['duration'].min()), int(df['duration'].max()), int(df['duration'].median()))
purpose = st.sidebar.selectbox("Purpose of Loan", df['purpose'].unique())

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

# ==================== Feature Engineering ====================
input_df['credit_per_month'] = input_df['credit_amount'] / (input_df['duration'] + 1e-3)
input_df['age_group'] = pd.cut(
    input_df['age'], bins=[18, 30, 40, 50, 60, 100],
    labels=['18-30', '31-40', '41-50', '51-60', '60+']
)
input_df['credit_to_age_ratio'] = input_df['credit_amount'] / input_df['age']
input_df['saving_level'] = input_df['saving_accounts'].map({'none': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4})
input_df['checking_level'] = input_df['checking_account'].map({'none': 0, 'little': 1, 'moderate': 2, 'rich': 3})
input_df['financial_security'] = input_df['saving_level'] / (input_df['credit_amount'] / 1000 + 1)

# ==================== Prediction ====================
X_proc = preprocessor.transform(input_df)
probas = model.predict_proba(X_proc)[0]
proba_bad = probas[0]
proba_good = probas[1]

threshold = st.sidebar.slider(
    "Bad-risk Probability Threshold (Adjust as Needed)",
    0.0, 1.0, 0.5, step=0.01
)

pred_label = 'bad' if proba_bad >= threshold else 'good'

# ==================== Main Page Layout ====================
tabs = st.tabs(["🔮 Prediction", "📊 Data Insights", "📈 Model Performance", "🧠 Model Explainability"])

# --- Prediction Tab ---
with tabs[0]:
    st.header("🔮 Prediction Result")
    if pred_label == 'bad':
        st.error("🔴 Predicted Risk: **Bad Credit**")
    else:
        st.success("🟢 Predicted Risk: **Good Credit**")

    st.markdown(f"### Probability of Bad Risk: **{proba_bad:.2%}**")
    st.info(" Tip: Use the sidebar threshold slider to adjust risk sensitivity for stricter or more lenient approval.")

# --- Data Insights Tab ---
with tabs[1]:
    st.header("📊 Customer Data Insights")
    col = st.selectbox("Select Feature to Explore", ['age', 'credit_amount', 'duration', 'financial_security'])
    fig = px.histogram(
        df, x=col, color='risk', barmode='overlay', marginal='box',
        title=f'Distribution of {col.capitalize()} by Credit Risk'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Model Performance Tab ---
with tabs[2]:
    st.header("📈 Model Performance Overview")
    st.subheader("Confusion Matrix")
    def load_plot(name, caption=None):
        path = os.path.join(PLOT_PATH, name)
        if os.path.exists(path):
            img = Image.open(path)
            st.image(img, caption=caption)
        else:
            st.warning(f" Plot `{name}` not found.")

    load_plot('stacking_ensemble_cm.png', "Confusion Matrix")
    
    st.subheader("ROC Curve")
    load_plot('stacking_ensemble_roc.png', "ROC AUC Curve")

# --- Model Interpretation Tab ---
with tabs[3]:
    st.header("🧠 Model Explainability (SHAP Analysis)")
    st.write("Understanding **which features** impact the prediction most:")
    
    st.subheader("Top Feature Importance (Bar Chart)")
    load_plot('stacking_ensemble_shap_summary_bar.png', "SHAP Feature Importance")
    
    st.subheader("Overall Feature Contribution (Summary Plot)")
    load_plot('stacking_ensemble_shap_summary.png', "SHAP Summary Plot")

# ==================== Footer ====================
st.write("---")
st.markdown("""
Designed  by **Krishnopreya**  
 📧 [Contact Developer](mailto:krish6.ch@gmail.com)
""")

