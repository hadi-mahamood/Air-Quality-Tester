import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model & encoder
# -----------------------------
model = joblib.load("air_quality_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Air Quality Predictor", page_icon="🌍")

st.title("🌍 Air Quality Prediction System")
st.write("Enter air pollutant values to predict AQI category.")

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 25.0)
    pm10 = st.number_input("PM10 (µg/m³)", 0.0, 500.0, 50.0)
    co = st.number_input("CO (ppm)", 0.0, 20.0, 1.0)
    no2 = st.number_input("NO₂ (ppb)", 0.0, 300.0, 30.0)

with col2:
    so2 = st.number_input("SO₂ (ppb)", 0.0, 300.0, 20.0)
    o3 = st.number_input("O₃ (ppb)", 0.0, 300.0, 40.0)
    temperature = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)

st.markdown("---")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict AQI"):

    input_data = pd.DataFrame([{
        "PM25": pm25,
        "PM10": pm10,
        "CO": co,
        "NO2": no2,
        "SO2": so2,
        "O3": o3,
        "Temperature": temperature,
        "Humidity": humidity
    }])

    prediction = model.predict(input_data)
    label = encoder.inverse_transform(prediction)[0]

    st.subheader("Prediction Result")

    if label == "Good":
        st.success(f"Air Quality: {label}")
    elif label == "Moderate":
        st.info(f"Air Quality: {label}")
    elif "Sensitive" in label:
        st.warning(f"Air Quality: {label}")
    elif label == "Unhealthy":
        st.error(f"Air Quality: {label}")
    else:
        st.error(f"Air Quality: {label}")

st.markdown("---")
st.caption("Air Quality Prediction Model | Machine Learning Project")
