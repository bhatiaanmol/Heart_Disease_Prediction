import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)


model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
st.title("❤️ Heart Disease Prediction")
st.write("Enter patient clinical details below to assess heart disease risk.")

st.markdown("---")

st.markdown("### 🧑 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol (mg/dl)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    restecg = st.selectbox("Resting ECG Result (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression", value=1.0)
    slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed, 3 = Reversible)", [1, 2, 3])

sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

st.markdown("---")

if st.button("🔍 Predict Heart Disease Risk"):
    sample = np.array([[age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak,
                        slope, ca, thal]])

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]

    st.markdown("### 🩺 Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ **High Risk of Heart Disease**\n\n"
            f"**Confidence:** {probability:.2%}"
        )
    else:
        st.success(
            f"✅ **Low Risk of Heart Disease**\n\n"
            f"**Confidence:** {(1 - probability):.2%}"
        )



