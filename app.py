import streamlit as st
import pandas as pd
import joblib

model = joblib.load("water_quality_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Water Quality Prediction")

pH = st.number_input("pH")
turbidity = st.number_input("Turbidity")
do = st.number_input("Dissolved Oxygen")
cond = st.number_input("Conductivity")
temp = st.number_input("Temperature")
tds = st.number_input("TDS")

if st.button("Predict"):
    data = pd.DataFrame([{
        "pH": pH,
        "Turbidity": turbidity,
        "Dissolved_Oxygen": do,
        "Conductivity": cond,
        "Temperature": temp,
        "TDS": tds
    }])

    pred = model.predict(data)
    label = encoder.inverse_transform(pred)
    st.success(f"Water Quality: {label[0]}")
