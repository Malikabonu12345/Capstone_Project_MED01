import pickle
import pandas as pd
import streamlit as st

# Load the trained model
with open("noshow_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Appointment No-Show Risk Predictor")

# Adjust these inputs to match the exact features your model was trained on
age = st.number_input("Patient age", min_value=0, max_value=115, value=30)
wait_days = st.number_input("Days between scheduling and appointment (WaitDays)", min_value=0, value=5)
sms_received = st.selectbox("SMS reminder sent?", ["No", "Yes"])
scholarship = st.selectbox("On welfare/scholarship program?", ["No", "Yes"])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "WaitDays": wait_days,
        "SMS_received": 1 if sms_received == "Yes" else 0,
        "Scholarship": 1 if scholarship == "Yes" else 0,
    }])
    prob = model.predict_proba(input_df)[0][1]
    st.metric("No-show risk", f"{prob:.0%}")
    if prob > 0.5:
        st.warning("High risk — recommend a reminder call.")
    else:
        st.success("Low risk.")
      streamlit
scikit-learn
pandas
pip install -r requirements.txt
streamlit run app.py
git add app.py requirements.txt
git commit -m "Add Streamlit demo app"
git push
