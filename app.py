import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
st.set_page_config(page_title="Loan Default Risk Dashboard", layout="wide")
st.title("📊 Loan Default Risk Dashboard")
# --- Load Model & Dataset ---
# NEW
@st.cache_resource
def load_model():
    return load("best_gbm_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("LoansData_sample.csv")

model = load_model()
data = load_data()

# --- Page Title ---



# --- Input Form ---
st.subheader("📝 Enter Loan Details")

col1, col2 = st.columns(2)
with col1:
    loan_amnt = st.number_input("Loan Amount", min_value=0)
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0)
    installment = st.number_input("Installment", min_value=0.0)
    annual_inc = st.number_input("Annual Income", min_value=0)

with col2:
    term = st.selectbox("Term", ["36 months", "60 months"])
    grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    home_ownership = st.selectbox("Home Ownership", ['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
    purpose = st.selectbox("Purpose", data['purpose'].dropna().unique())

# --- Prediction Button ---
st.markdown("---")
if st.button("🔍 Predict Default Risk"):
    input_data = pd.DataFrame({
        "loan_amnt": [loan_amnt],
        "int_rate": [int_rate],
        "installment": [installment],
        "annual_inc": [annual_inc],
        "term": [term],
        "grade": [grade],
        "home_ownership": [home_ownership],
        "purpose": [purpose]
    })

    # Preprocessing (ensure it matches model training!)
    input_data_encoded = pd.get_dummies(input_data)
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0
    input_data_encoded = input_data_encoded[model_features]

    prediction = model.predict(input_data_encoded)[0]
    prob = model.predict_proba(input_data_encoded)[0][1]

    st.metric("Probability of Default", f"{prob*100:.2f} %")
    if prediction == 1:
        st.error("🔴 High Risk of Default")
    else:
        st.success("🟢 Low Risk of Default")

# --- Tabs for Visuals ---
tab1, tab2= st.tabs(["📈 Visualizations", "📁 Dataset Summary"])

with tab1:
    st.subheader("Annual Income Distribution")
    st.bar_chart(data['annual_inc'].sample(100))

    st.subheader("Loan Purpose Distribution")
    st.bar_chart(data['purpose'].value_counts())

    st.subheader("Grade Distribution")
    st.bar_chart(data['grade'].value_counts())

with tab2:
    st.subheader("Dataset Overview")
    st.write("Shape:", data.shape)
    st.dataframe(data.head())

    st.subheader("Statistics Summary")
    st.dataframe(data.describe())


# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Model: `best_gbm_model.joblib` | Data: `LoansData_sample.csv`")

