import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Health Insurance Premium Predictor", layout="centered")

st.title("🛡️ Shield Insurance Premium Predictor")

# Load model bundle (model + scaler + poly)
@st.cache_data
def load_model():
    model_path = os.path.join("models", "insurance_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

bundle = load_model()
model = bundle['model']
scaler = bundle['scaler']
poly = bundle['poly']

# Input fields for user
age = st.slider("Age", min_value=18, max_value=100, value=30)
bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoker = st.radio("Smoker", options=["No", "Yes"])
children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])
sex = st.radio("Sex", options=["Male", "Female"])

# Encode categorical inputs
smoker_val = 1 if smoker == "Yes" else 0
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region_val = region_map[region]
sex_val = 0 if sex == "Male" else 1

# Prepare input vector
input_array = np.array([[age, bmi, smoker_val, children, region_val, sex_val]])

# Transform input features
input_poly = poly.transform(input_array)
input_scaled = scaler.transform(input_poly)

# Predict on button click
if st.button("Predict Premium"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Annual Health Insurance Premium: ${prediction:,.2f}")

st.markdown("---")
st.markdown("Developed by AtliQ AI for Shield Insurance")
