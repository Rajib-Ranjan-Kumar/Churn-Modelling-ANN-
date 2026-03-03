import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}

/* Title */
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: white;
    padding: 20px;
}

/* Card */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
}

/* Result box */
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    margin-top: 20px;
}

/* Success */
.success {
    background-color: #d4edda;
    color: #155724;
}

/* Danger */
.danger {
    background-color: #f8d7da;
    color: #721c24;
}

/* Button */
.stButton>button {
    background-color: #1f4037;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    border: none;
}

.stButton>button:hover {
    background-color: #14532d;
}

/* Dropdown Styling */
div[data-baseweb="select"] > div {
    border-radius: 10px;
    border: 1px solid #1f4037;
}

/* Number Input */
input {
    border-radius: 8px !important;
    border: 1px solid #1f4037 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_resource
def load_encoders():
    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return label_encoder_gender, onehot_encoder_geo, scaler


model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_encoders()

# ---------------- TITLE ----------------
st.markdown('<div class="title">📊 Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)

# ---------------- CARD ----------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
        gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
        age = st.slider("🎂 Age", 18, 92, 30)
        tenure = st.slider("📅 Tenure (Years)", 0, 10, 3)
        num_of_products = st.slider("📦 Number of Products", 1, 4, 1)

    with col2:
        balance = st.number_input("💰 Balance", min_value=0.0, value=50000.0)
        credit_score = st.number_input("🏦 Credit Score", min_value=300, max_value=900, value=600)
        estimated_salary = st.number_input("💵 Estimated Salary", min_value=0.0, value=50000.0)
        has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
        is_active_member = st.selectbox("⭐ Is Active Member", [0, 1])

    predict_button = st.button("🚀 Predict Churn")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_button:

    with st.spinner("Analyzing customer data..."):

        # Create DataFrame
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [label_encoder_gender.transform([gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        })

        # Encode Geography
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
        )

        # Combine data
        input_data = pd.concat([input_data, geo_encoded_df], axis=1)

        # Scale
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data_scaled)
        prediction_proba = float(prediction[0][0])

    # ---------------- RESULT ----------------
    st.subheader("Prediction Result")

    if prediction_proba > 0.5:
        st.markdown(
            f'<div class="result-box danger">⚠️ Churn Probability: {prediction_proba:.2f}<br>Customer is likely to churn.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box success">✅ Churn Probability: {prediction_proba:.2f}<br>Customer is not likely to churn.</div>',
            unsafe_allow_html=True
        )