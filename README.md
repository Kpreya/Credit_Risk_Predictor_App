# Credit Risk Predictor

## Overview
This application predicts credit risk using machine learning algorithms applied to German credit data. It features a comprehensive preprocessing pipeline, multiple trained models including Random Forest, Logistic Regression, SVC, and XGBoost, with a stacking ensemble for optimal performance.

[Live Demo]: (https://creditriskpredictor.streamlit.app/)

## Features
- **Interactive Web Interface**: User-friendly Streamlit application for real-time credit risk predictions
- **Advanced Preprocessing**: Feature engineering, outlier handling, and data imputation techniques
- **Ensemble Modeling**: Stacking ensemble approach combining multiple ML algorithms
- **Model Interpretation**: SHAP analysis for explainable AI
- **Performance Metrics**: Confusion matrices and ROC curves for transparent model evaluation

## Tech Stack
- **Python**: Core programming language
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting implementation
- **SMOTE**: Handling class imbalance
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Plotly**: Data visualization
- **SHAP**: Model interpretation

## Project Structure
```
.
├── creditrisk.py              # Preprocessing and feature engineering
├── train_models.py            # Model training script
├── shap_analysis.py           # Model interpretation with SHAP
├── performance_plots.py       # Generate performance visualizations
├── credit_risk_streamlitapp.py # Web application
├── data/
│   ├── german_credit_data.csv  # Original dataset
│   └── preprocessed_credit_data.csv # Preprocessed data
├── models/                    # Saved models and encoders
└── plots/                     # Performance and interpretation plots
```

## Installation
```bash
# Clone the repository
git clone https://github.com/Kpreya/Credit_Risk_Predictor_App.git
cd Credit_Risk_Predictor_App

# Install dependencies
pip install -r requirements.txt
```

## Usage
### 1. Data Preprocessing
```bash
python creditrisk.py
```
This script loads the German credit dataset, performs cleaning, feature engineering, and preprocessing tasks, and saves the artifacts for model training.

### 2. Model Training
```bash
python train_models.py
```
Trains multiple machine learning models and saves them for later use.

### 3. Model Analysis
```bash
python shap_analysis.py
python performance_plots.py
```
Generates model interpretation plots and performance metrics.

### 4. Running the Web App
```bash
streamlit run credit_risk_streamlitapp.py
```
Launches the interactive web application for credit risk prediction.

## Model Performance
The application employs a stacking ensemble of multiple base models to achieve optimal prediction performance. Model evaluation metrics include:
- Confusion matrix
- ROC curve
- SHAP feature importance
- Cross-validation scores

## Feature Importance
Key factors in credit risk prediction include:
- Financial security (saving level relative to credit amount)
- Checking account status
- Credit amount and duration
- Savings account level
- Purpose of the loan
- Age and employment status

## Future Enhancements
- Additional model algorithms for performance comparison
- Deep learning approaches for complex pattern recognition
- API deployment for integration with banking systems
- Enhanced visualization options for better interpretability

## License
[MIT](LICENSE)

## Author
Developed by Krishnopreya C.
Contact :krish6.ch@gmail.com

## Acknowledgments
- German Credit Data from UCI Machine Learning Repository
- SHAP library for model interpretation
- Streamlit for web application framework
