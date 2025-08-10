
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('best_heart_disease_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    return model, scaler

def main():
    st.title("Heart Disease Prediction App")
    st.write("Enter patient information to predict heart disease risk")

    # Load model
    model, scaler = load_model()

    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], 
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x-1])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 250)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                          format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG", [0, 1, 2], 
                              format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])

    with col2:
        thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], 
                            format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.slider("ST Depression", 0.0, 7.0, 1.0, 0.1)
        slope = st.selectbox("ST Slope", [1, 2, 3], 
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x-1])
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [3, 6, 7], 
                           format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x//3-1])

    # Create prediction button
    if st.button("Predict Heart Disease Risk"):
        # Prepare input data
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                               thalach, exang, oldpeak, slope, ca, thal]])

        # Scale the input if needed
        if hasattr(model, 'predict_proba'):
            # Check if model needs scaling
            model_name = str(type(model).__name__)
            if model_name in ['LogisticRegression', 'SVC']:
                input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # Display results
        st.markdown("---")
        if prediction == 1:
            st.error("HIGH RISK: Heart disease detected!")
            st.write(f"Probability of heart disease: {probability[1]:.2%}")
        else:
            st.success("LOW RISK: No heart disease detected!")
            st.write(f"Probability of no heart disease: {probability[0]:.2%}")

        # Risk factors analysis
        st.markdown("### Risk Factors Analysis")
        risk_factors = []

        if age > 55:
            risk_factors.append("Advanced age")
        if sex == 1:
            risk_factors.append("Male gender")
        if trestbps > 140:
            risk_factors.append("High blood pressure")
        if chol > 240:
            risk_factors.append("High cholesterol")
        if thalach < 100:
            risk_factors.append("Low maximum heart rate")
        if exang == 1:
            risk_factors.append("Exercise induced angina")

        if risk_factors:
            st.write("Identified risk factors:")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("No major risk factors identified.")

if __name__ == "__main__":
    main()
