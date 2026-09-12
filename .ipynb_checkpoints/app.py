import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# ---------------------------------------------------
# Load trained pipeline
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")


model = load_model()


# ---------------------------------------------------
# Title and description
# ---------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    """
    This application predicts whether a customer is likely to churn
    based on customer demographic, service, billing and usage information.

    Enter the customer information below and click **Predict Churn**.
    """
)


# ---------------------------------------------------
# Customer information
# ---------------------------------------------------

st.header("Customer Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

region = st.selectbox(
    "Region",
    ["East", "West", "North", "South"]
)

tenure_months = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=120,
    value=12,
    step=1
)


# ---------------------------------------------------
# Billing information
# ---------------------------------------------------

st.header("Billing Information")

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=1000.0,
    value=70.0,
    step=1.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=100000.0,
    value=840.0,
    step=10.0
)

contract_type = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Bank transfer",
        "Credit card",
        "Mailed check"
    ]
)


# ---------------------------------------------------
# Internet/service information
# ---------------------------------------------------

st.header("Internet & Support Information")

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)


# ---------------------------------------------------
# Usage and support information
# ---------------------------------------------------

st.header("Usage & Customer Support")

num_support_calls = st.number_input(
    "Number of Support Calls",
    min_value=0,
    max_value=100,
    value=2,
    step=1
)

late_payments_last_year = st.number_input(
    "Late Payments Last Year",
    min_value=0,
    max_value=50,
    value=1,
    step=1
)

avg_monthly_usage_gb = st.number_input(
    "Average Monthly Usage (GB)",
    min_value=0.0,
    max_value=5000.0,
    value=200.0,
    step=10.0
)


# ---------------------------------------------------
# Prediction button
# ---------------------------------------------------

if st.button("Predict Churn"):

    try:

        # Create dataframe with exactly the same feature names
        # used during model training.

        input_data = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "region": [region],
            "tenure_months": [tenure_months],
            "monthly_charges": [monthly_charges],
            "total_charges": [total_charges],
            "contract_type": [contract_type],
            "internet_service": [internet_service],
            "tech_support": [tech_support],
            "online_security": [online_security],
            "paperless_billing": [paperless_billing],
            "payment_method": [payment_method],
            "num_support_calls": [num_support_calls],
            "late_payments_last_year": [late_payments_last_year],
            "avg_monthly_usage_gb": [avg_monthly_usage_gb]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display result
        if prediction == 1:

            st.error("⚠️ Prediction: Yes — Customer is likely to churn.")

        else:

            st.success("✅ Prediction: No — Customer is unlikely to churn.")


        # ---------------------------------------------------
        # Prediction probability
        # ---------------------------------------------------

        probabilities = model.predict_proba(input_data)[0]

        churn_probability = probabilities[1]

        st.write(
            f"Estimated churn probability: "
            f"**{churn_probability:.2%}**"
        )

    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.write(str(e))