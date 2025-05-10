
import streamlit as st
import pandas as pd
import joblib

# Load model, label encoder, and feature list
model = joblib.load("accident_risk_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("AI-Driven Traffic Accident Risk Predictor")

# Define form for input
st.markdown("### Enter Traffic Scenario Details:")

user_input = {}
for feature in feature_columns:
    user_input[feature] = st.text_input(f"{feature.replace('_', ' ')}")

# Predict button
if st.button("Predict Accident Severity"):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Encode categorical features using the same label encoder
        for col in input_df.columns:
            try:
                input_df[col] = label_encoder.transform(input_df[col])
            except:
                pass  # assume it's numeric or already encoded

        # Match column order
        input_df = input_df[feature_columns]

        # Predict
        prediction = model.predict(input_df)[0]
        severity_map = {0: "Minor", 1: "Moderate", 2: "Severe"}
        st.success(f"Predicted Accident Severity: {severity_map.get(prediction, prediction)}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
