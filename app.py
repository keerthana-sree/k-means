import streamlit as st
import joblib
import numpy as np

# Load scaler & model (joblib)
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.set_page_config(page_title="Customer Segmentation App", page_icon="📊")

st.title("📊 Customer Segmentation (K-Means)")
st.write("Enter customer details to find cluster")

# Inputs
income = st.number_input("Annual Income (k$)", min_value=0.0, max_value=200.0, value=50.0)
spending = st.number_input("Spending Score (1-100)", min_value=0.0, max_value=100.0, value=50.0)

if st.button("Predict Cluster"):
    data = np.array([[income, spending]])
    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)

    st.success(f"✅ Customer belongs to Cluster: {cluster[0]}")
