import joblib
import pandas as pd
import streamlit as st

model = joblib.load("noshow_model.pkl")

st.title("Appointment No-Show Risk Predictor")

age = st.number_input("Patient age", min_value=0, max_value=115, value=30)
scholarship = st.selectbox("On welfare/scholarship program?", ["No", "Yes"])
hypertension = st.selectbox("Has hypertension?", ["No", "Yes"])
diabetes = st.selectbox("Has diabetes?", ["No", "Yes"])
alcoholism = st.selectbox("Alcoholism?", ["No", "Yes"])
handicap = st.selectbox("Handicap?", ["No", "Yes"])
sms_received = st.selectbox("SMS reminder sent?", ["No", "Yes"])
wait_days = st.number_input("Days between scheduling and appointment (WaitDays)", min_value=0, value=5)
gender_m = st.selectbox("Gender", ["Female", "Male"])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Scholarship": 1 if scholarship == "Yes" else 0,
        "Hypertension": 1 if hypertension == "Yes" else 0,
        "Diabetes": 1 if diabetes == "Yes" else 0,
        "Alcoholism": 1 if alcoholism == "Yes" else 0,
        "Handicap": 1 if handicap == "Yes" else 0,
        "SMS_received": 1 if sms_received == "Yes" else 0,
        "WaitDays": wait_days,
        "Gender_M": 1 if gender_m == "Male" else 0,
    }])
    prob = model.predict_proba(input_df)[0][1]
    st.metric("No-show risk", f"{prob:.0%}")
    if prob > 0.5:
        st.warning("High risk — recommend a reminder call.")
    else:
        st.success("Low risk.")
