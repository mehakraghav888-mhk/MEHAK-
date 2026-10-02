import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to estimate the probability that the transaction is potentially fraudulent.")

if not MODEL_PATH.exists():
    st.error("Model not found. First run: python src/generate_dataset.py and python src/train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

amount = st.number_input("Transaction amount", min_value=1.0, max_value=5000.0, value=250.0, step=10.0)
hour = st.slider("Transaction hour (0–23)", 0, 23, 14)
distance = st.number_input("Distance from usual location (km)", min_value=0.0, max_value=500.0, value=5.0)
velocity = st.number_input("Transactions in last hour", min_value=0.0, max_value=20.0, value=1.0)
device_change = st.selectbox("New / changed device?", ["No", "Yes"])
international = st.selectbox("International transaction?", ["No", "Yes"])
failed_attempts = st.number_input("Failed attempts before transaction", min_value=0, max_value=10, value=0)
merchant_risk = st.slider("Merchant risk score", 0.0, 1.0, 0.20, 0.01)

row = pd.DataFrame([{
    "amount": amount,
    "hour": hour,
    "distance_km": distance,
    "transactions_last_hour": velocity,
    "device_change": 1 if device_change == "Yes" else 0,
    "international": 1 if international == "Yes" else 0,
    "failed_attempts": failed_attempts,
    "merchant_risk": merchant_risk
}])

if st.button("Check Transaction"):
    fraud_probability = float(model.predict_proba(row)[0, 1])
    prediction = int(model.predict(row)[0])

    st.metric("Estimated fraud probability", f"{fraud_probability * 100:.2f}%")
    if prediction == 1:
        st.error("⚠️ Potentially suspicious transaction")
    else:
        st.success("✅ Transaction classified as non-fraudulent")

st.caption("Educational demonstration only. This prediction should not be used as the sole basis for real financial decisions.")
