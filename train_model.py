import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import xgboost as xgb
import lightgbm as lgb

# Load dataset
df = pd.read_csv("data/insurance.csv")

# Encode categorical variables
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df["region"] = df["region"].map({"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3})

# Base features
X_base = df[["age", "bmi", "smoker", "children", "region", "sex"]]
y = df["charges"]

# Polynomial Features degree 2 (interactions + squares)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_base)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Scaling for polynomial features (important!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base models
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
xgbr = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgbm = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

# Stacking regressor with Linear Regression as final estimator
from sklearn.linear_model import LinearRegression

stack = StackingRegressor(
    estimators=[('rf', rf), ('xgb', xgbr), ('lgbm', lgbm)],
    final_estimator=LinearRegression(),
    n_jobs=-1,
    passthrough=True
)

# Train stacking model
stack.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = stack.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
within_10pct = np.mean(np.abs((y_pred - y_test) / y_test) < 0.10)

print("=== Model Evaluation ===")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"% predictions within 10% error: {within_10pct * 100:.2f}%")

# Save model & scaler & poly transformer together as a dict
model_bundle = {
    'model': stack,
    'scaler': scaler,
    'poly': poly
}

os.makedirs("models", exist_ok=True)
with open("models/insurance_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("Model saved at models/insurance_model.pkl")
