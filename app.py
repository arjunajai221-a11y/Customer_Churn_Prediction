
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Analyze customer behavior and predict customer churn using Machine Learning.")

# =====================================
# 2. LOAD DATA AND MODEL
# =====================================

@st.cache_resource
def load_model():
    return joblib.load("best_churn_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

model = load_model()
df = load_data()

# =====================================
# 3. DATA OVERVIEW
# =====================================

total_customers = len(df)
churned_customers = (df["Churn"] == "Yes").sum()
retained_customers = (df["Churn"] == "No").sum()
churn_rate = (churned_customers / total_customers) * 100

st.subheader("📌 Business Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Retained Customers", f"{retained_customers:,}")
col4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# =====================================
# 4. DATA VISUALIZATION
# =====================================

st.subheader("📈 Customer Analytics")

chart1, chart2 = st.columns(2)

with chart1:
    churn_counts = df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Customers"]

    fig = px.pie(
        churn_counts,
        names="Churn",
        values="Customers",
        title="Customer Churn Distribution",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

with chart2:
    contract_data = (
        df.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        contract_data,
        x="Contract",
        y="Customers",
        color="Churn",
        barmode="group",
        title="Contract Type vs Churn"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================
# 5. CUSTOMER PREDICTION
# =====================================

st.divider()
st.subheader("🔍 Predict Customer Churn")

st.write("Enter customer details below to predict whether the customer is likely to leave.")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            0, 72, 12
        )

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=500.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=monthly_charges * tenure
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

# =====================================
# 6. DISPLAY PREDICTION
# =====================================

if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.error("⚠️ Customer is likely to Churn")
        else:
            st.success("✅ Customer is likely to Stay")

    with result_col2:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    st.progress(float(probability))

    if prediction == 1:
        st.warning(
            "Recommendation: Consider offering customer support, "
            "loyalty benefits, or suitable retention plans."
        )
    else:
        st.info(
            "Recommendation: Continue providing good customer service "
            "to maintain customer satisfaction."
        )

# =====================================
# 7. FOOTER
# =====================================

st.divider()

st.caption(
    "Customer Churn Prediction | Python | Machine Learning | Streamlit"
)
st.caption("Developed by Arjun Ajai")
