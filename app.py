# app.py
import streamlit as st
import pandas as pd
import numpy as np
from model import train_model
from utils import load_and_validate_csv

st.set_page_config(page_title="🔬 Breast Cancer Detection", layout="centered")

# Main Title and Description
st.markdown("## 🔬 Breast Cancer Detection App")
st.markdown("""
Upload a CSV dataset, train a model, and predict if a tumor is **Malignant (M)** or **Benign (B)**.
""")

# File Upload
st.markdown("### 📂 Upload your breast cancer CSV file")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Upload Breast Cancer Wisconsin Diagnostic CSV", label_visibility="collapsed")

if uploaded_file is not None:
    # Load and validate CSV
    df, error = load_and_validate_csv(uploaded_file)

    if error:
        st.error(f"❌ File Error: {error}")
    else:
        st.success("✅ File successfully loaded and validated!")
        
        with st.spinner("Training model..."):
            model, accuracy, feature_list = train_model(df)

        st.success(f"✅ Model trained with **{accuracy*100:.2f}% accuracy**")

        # Prediction Form
        st.markdown("### 📋 Enter Patient Features")

        input_data = []
        for col in feature_list:
            val = st.slider(
                label=f"{col.replace('_', ' ').capitalize()}",
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].mean()),
                step=0.01
            )
            input_data.append(val)

        # Predict button
        if st.button("🔍 Predict Tumor Type"):
            pred = model.predict(np.array(input_data).reshape(1, -1))[0]
            result = "🟢 **Benign**" if pred == 1 else "🔴 **Malignant**"
            st.markdown(f"### 🧪 Prediction Result: {result}")
else:
    st.info("📁 Awaiting CSV file upload...")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by [Your Name]")
