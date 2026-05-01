import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

st.title("Advanced Network Intrusion Detection System")
st.markdown("NSL-KDD Explainable IDS Dashboard")

model = joblib.load("models/model.pkl")

uploaded_file = st.file_uploader("Upload Processed CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "label" in df.columns:
        df = df.drop("label", axis=1)

    threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5)

    probabilities = model.predict_proba(df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    df["Risk Score"] = probabilities
    df["Prediction"] = predictions

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Samples", len(df))

    with col2:
        st.metric("Attacks Detected", int(sum(predictions)))

    with col3:
        st.metric("Normal Traffic", int(len(df) - sum(predictions)))

    st.subheader("Prediction Preview")
    st.dataframe(df.head())

    # Pie Chart
    st.subheader("Attack Distribution")
    fig1, ax1 = plt.subplots()
    ax1.pie(
        [sum(predictions), len(df)-sum(predictions)],
        labels=["Attack", "Normal"],
        autopct="%1.1f%%"
    )
    st.pyplot(fig1)

    # Risk Distribution
    st.subheader("Risk Score Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(probabilities, bins=30)
    st.pyplot(fig2)

    # SHAP Global Explanation
    st.subheader("Global Feature Importance (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df.drop(["Risk Score","Prediction"], axis=1))

    fig3 = plt.figure()
    shap.summary_plot(shap_values, df.drop(["Risk Score","Prediction"], axis=1), show=False)
    st.pyplot(fig3)

    # Row-level explanation
    st.subheader("Local Explanation for Selected Row")
    row_index = st.number_input("Select Row Index", 0, len(df)-1, 0)

    fig4 = plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[row_index],
        df.drop(["Risk Score","Prediction"], axis=1).iloc[row_index],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig4)

    # Download predictions
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )