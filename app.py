import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load saved "brains"
model = joblib.load('carepredict_model.pkl')
scaler = joblib.load('hospital_scaler.pkl')
features = joblib.load('feature_metadata.pkl')

# 2. Risk classifications


def get_risk_level(probability):
    prob_percent = probability*100
    if prob_percent < 20:
        return 'Low Risk', 'green'
    elif prob_percent < 45:
        return 'Medium Risk', 'yellow'
    elif prob_percent < 75:
        return 'High Risk', 'orange'
    else:
        return 'EMERGENCY / CRITICAL RISK', 'red'


# 3. UI Header
st.set_page_config(page_title='CarePredict System', layout='wide')
st.title('Hospital Readmission Risk Prediction System')
st.markdown('---')

# Sidebar for users
st.sidebar.header('Patient Clinical Data')
age_val = st.sidebar.slider('Patient Age', 0, 100, 55)
time_val = st.sidebar.slider('Time in Hospital (Days)', 1, 14, 3)
lab_val = st.sidebar.number_input(
    'Number of Lab Procedures', 1, 150, 45)
med_val = st.sidebar.number_input('Number of Medications', 1, 100, 15)
prev_inpatient = st.sidebar.slider(
    'Previous Inpatient Visits (Last Year)', 0, 20, 0)
prev_emergency = st.sidebar.slider(
    'Previous Emergency Visits (Last Year)', 0, 20, 0)
num_diagnoses = st.sidebar.slider('Number of Registered Diagnosis', 1, 16, 5)


if st.sidebar.button("Run Risk Analysis"):
    # 1. Create template
    input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    # 2. Map UI values to model columns
    input_data['age'] = age_val
    input_data['time_in_hospital'] = time_val
    input_data['num_lab_procedures'] = lab_val
    input_data['num_medications'] = med_val
    input_data['number_inpatient'] = prev_inpatient
    input_data['number_emergency'] = prev_emergency
    input_data['number_diagnoses'] = num_diagnoses
    # 3. Process and Predict
    scaled_data = scaler.transform(input_data)
    prediction_prob = model.predict_proba(scaled_data)[0][1]
    risk_text, risk_color = get_risk_level(prediction_prob)
    # 4. Display Results
    st.markdown(f'### Analysis Result')
    st.subheader(f'Risk Level: :{risk_color}[{risk_text}]')
    st.write(f'Probability Score: **{prediction_prob:.2%}**')
    if prediction_prob > 0.45:
        st.warning(
            'Action Required = Patient shows high markers for potental readmission')
    else:
        st.success('Patient currently withing stable post-discharge conditions')
# Run with "streamlit run app.py" in terminal
