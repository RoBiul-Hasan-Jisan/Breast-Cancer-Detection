# health_insurance_predictor
# Predictive Health Insurance Model for Shield Insurance

## Overview

This project develops a predictive model to estimate health insurance premiums based on user factors such as age, BMI, smoker status, number of children, region, and sex. The goal is to help insurance underwriters quickly generate accurate premium quotes using a simple and interactive web application.

---

## Project Structure
health_insurance_predictor/
│
├── data/
│ └── insurance.csv #
│
├── models/
│ └── insurance_model.pkl # Trained model bundle (stacking ensemble + scaler  polynomial features)
│
├── train_model.py # Script to train and save the model
│
├── app.py # Streamlit app for interactive prediction
│
└── README.md 

---

## Installation

1. Clone the repository:
   ```bash
   git clone <your_repo_url>
   cd health_insurance_predictor
   
pip install -r requirements.txt
pip install pandas numpy scikit-learn xgboost lightgbm streamlit


Model Details
Model Type: Stacking ensemble of Random Forest, XGBoost, and LightGBM regressors.

Features Used:
age, bmi, smoker, children, region, sex
plus polynomial interaction features (degree 2).

Data Processing:
Categorical encoding, polynomial feature expansion, feature scaling with StandardScaler.

Performance:
Typical R² around 0.87-0.88 on test data with current dataset and features.

Limitations
Achieving >97% R² accuracy with this dataset and features alone is very challenging due to inherent data noise and variability.

Further improvement requires richer data, advanced feature engineering, or more complex models.
