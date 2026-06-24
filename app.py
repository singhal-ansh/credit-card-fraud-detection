import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
lr_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detector")
st.markdown("Enter transaction details to predict whether it's fraudulent.")

# Model selector
model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression (High Recall)", "Random Forest (High Precision)"]
)

st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=50000.0)

st.markdown("**PCA Features (V1–V14)**")
cols = st.columns(4)
v_features = {}
for i in range(1, 15):
    col_idx = (i - 1) % 4
    with cols[col_idx]:
        v_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f")

st.markdown("**PCA Features (V15–V28)**")
cols2 = st.columns(4)
for i in range(15, 29):
    col_idx = (i - 15) % 4
    with cols2[col_idx]:
        v_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f")

if st.button("🔍 Predict"):
    # Scale amount and time
    amount_scaled = scaler.transform([[amount]])[0][0]
    time_scaled = (time - 94813) / 47488  # approximate normalization

    # Build input
    input_data = {**v_features, "Amount_scaled": amount_scaled, "Time_scaled": time_scaled}
    input_df = pd.DataFrame([input_data])

    # Predict
    model = lr_model if "Logistic" in model_choice else rf_model
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED — Confidence: {probability:.1%}")
    else:
        st.success(f"✅ LEGITIMATE — Fraud Probability: {probability:.1%}")

