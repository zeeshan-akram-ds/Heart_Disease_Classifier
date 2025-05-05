import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Load pre trained pipeline
model = joblib.load('rfc_heart_disease.pkl')  
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Risk Predictor")
st.markdown("Enter your medical details in the sidebar to get prediction for heart disease.")

# Sidebar inputs
st.sidebar.header("Patient Info")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
cp_type = st.sidebar.selectbox("Chest Pain Type", ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'])
rest_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar", ['Greater than 120 mg/ml', 'Lower than 120 mg/ml'])
rest_ecg = st.sidebar.selectbox("Resting ECG", ['Normal', 'Left ventricular hypertrophy', 'ST-T wave abnormality'])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
angina = st.sidebar.selectbox("Exercise-Induced Angina", ['Yes', 'No'])
oldpeak = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of ST Segment", ['Upsloping', 'Flat', 'Downsloping'])
vessels = st.sidebar.selectbox("No. of Vessels Colored", ['0', '1', '2', '3'])
thal = st.sidebar.selectbox("Thalassemia", ['Normal', 'Fixed defect', 'Reversible defect'])

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'chest_pain_type': cp_type,
        'resting_blood_pressure': rest_bp,
        'cholestoral': chol,
        'fasting_blood_sugar': fbs,
        'rest_ecg': rest_ecg,
        'Max_heart_rate': max_hr,
        'exercise_induced_angina': angina,
        'oldpeak': oldpeak,
        'slope': slope,
        'vessels_colored_by_flourosopy': vessels,
        'thalassemia': thal
    }])

    # engineering Features
    input_data['cholesterol_per_age'] = input_data['cholestoral'] / input_data['age']
    input_data['hr_age_ratio'] = input_data['Max_heart_rate'] / input_data['age']
    input_data['bp_cholesterol_interaction'] = input_data['resting_blood_pressure'] * input_data['cholestoral']
    input_data['ex_induced_pain_severity'] = input_data['exercise_induced_angina'].map({'Yes': 1, 'No': 0}) * input_data['oldpeak']
    input_data['heart_stress_score'] = input_data['resting_blood_pressure'] + input_data['cholestoral'] + input_data['oldpeak']

    probability = model.predict_proba(input_data)[0][1]
    predictions = model.predict(input_data)
    st.subheader("Prediction Result")
    if predictions == 1:
        st.error(f"**High Risk!**  \n**Probability: {probability:.2%}**")
    else:
        st.success(f"**Low Risk!**  \n**Probability: {probability:.2%}**")
    st.markdown("---")
    st.subheader("Model Evaluation (on test set)")
    with st.expander("Show Model Evaluation Metrics"):
        st.markdown("""
        - **Accuracy:** 98.05%  
        - **Precision (Class 0):** 0.96  
        - **Recall (Class 0):** 1.00  
        - **F1-score (Class 0):** 0.98  

        - **Precision (Class 1):** 1.00  
        - **Recall (Class 1):** 0.96  
        - **F1-score (Class 1):** 0.98  

        - **Confusion Matrix:**  
        ```
        [[150   0]
        [  6 152]]
        ```
        """)

    st.markdown(" *Note: This tool is for educational use. Always consult a medical professional.*")
