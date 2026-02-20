import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("aqi_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Air Quality Prediction System")
st.write("Enter air pollution values to predict AQI category")

# User inputs
pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0)
pm10 = st.number_input("PM10 (µg/m³)", 0.0, 500.0)
co2 = st.number_input("CO₂ (ppm)", 300.0, 2000.0)

if st.button("Predict AQI"):
    
    input_data = pd.DataFrame({
        "PM25_sensor": [pm25],
        "PM10_sensor": [pm10],
        "CO2_sensor": [co2]
    })

    prediction = model.predict(input_data)
    label = encoder.inverse_transform(prediction)

    st.success(f"Predicted AQI Category: {label[0]}")
