import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Student Score Predictor")
st.write("Enter the number of hours studied to predict the exam score.")

# Input: Hours studied
hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=0.1)

if st.button("Predict Score"):
    # Prepare input data as DataFrame (model expects DataFrame)
    input_df = pd.DataFrame({'Hours': [hours]})
    
    # Predict using the loaded model
    prediction = model.predict(input_df)
    
    # Show result
    st.success(f"Predicted Score: {prediction[0]:.2f}")
