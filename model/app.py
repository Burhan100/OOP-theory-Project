import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Page Title
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.write("Fill in the details below to predict the house price using a trained SVM model.")

# UI Form
with st.form("input_form"):
    area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)
    bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)
    stories = st.slider("Number of Stories", 1, 5, 1)
    mainroad = st.selectbox("Main Road Access", ['yes', 'no'])
    guestroom = st.selectbox("Guest Room Available", ['yes', 'no'])
    basement = st.selectbox("Basement Available", ['yes', 'no'])
    hotwaterheating = st.selectbox("Hot Water Heating", ['yes', 'no'])
    airconditioning = st.selectbox("Air Conditioning", ['yes', 'no'])
    parking = st.slider("Parking Spaces", 0, 5, 1)
    prefarea = st.selectbox("Preferred Area", ['yes', 'no'])
    furnishingstatus = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

    submit = st.form_submit_button("Predict Price")

# Encoding helper
def encode_input(area, bedrooms, bathrooms, stories, mainroad, guestroom,
                 basement, hotwaterheating, airconditioning, parking,
                 prefarea, furnishingstatus):

    data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad_yes': 1 if mainroad == 'yes' else 0,
        'guestroom_yes': 1 if guestroom == 'yes' else 0,
        'basement_yes': 1 if basement == 'yes' else 0,
        'hotwaterheating_yes': 1 if hotwaterheating == 'yes' else 0,
        'airconditioning_yes': 1 if airconditioning == 'yes' else 0,
        'prefarea_yes': 1 if prefarea == 'yes' else 0,
        'furnishingstatus_semi-furnished': 1 if furnishingstatus == 'semi-furnished' else 0,
        'furnishingstatus_unfurnished': 1 if furnishingstatus == 'unfurnished' else 0
    }

    return pd.DataFrame([data])

# Prediction
if submit:
    user_input = encode_input(area, bedrooms, bathrooms, stories, mainroad,
                              guestroom, basement, hotwaterheating,
                              airconditioning, parking, prefarea, furnishingstatus)
    
    # Scale and predict
    scaled_input = scaler.transform(user_input)
    predicted_price = model.predict(scaled_input)[0]

    st.success(f"💰 Estimated House Price: ₹ {predicted_price:,.2f}")
