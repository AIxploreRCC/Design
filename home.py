import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
import os

# Charger le modèle avec mise en cache
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'coxph_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load(model_path)

# Charger le modèle
model_cox = load_model()

def home():
    # Title of the Streamlit app
    st.title("Survival Prediction with the Cox Model")

    # Inputs for the model's variables
    HbN = st.selectbox("HbN", options=[0, 1])
    N = st.selectbox("N", options=[0, 1, 2])
    rad = st.slider("Radiomics Signature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    Thrombus = st.selectbox("Thrombus", options=[0, 1, 2, 3])

    # DataFrame for model input
    input_df = pd.DataFrame({
        'HbN': [HbN],
        'N': [N],
        'rad': [rad],
        'Thrombus': [Thrombus]
    })

    # Convert categorical variables to category type
    input_df['N'] = input_df['N'].astype('category')
    input_df['Thrombus'] = input_df['Thrombus'].astype('category')

    # Button to predict survival
    if st.button('Predict Survival'):
        with st.spinner('Calculating... Please wait.'):
            try:
                survival_function = model_cox.predict_survival_function(input_df)
                st.subheader('Estimated Survival Probability:')
                
                # Prepare data for plotting
                time_points = survival_function[0].x
                survival_probabilities = [fn(time_points) for fn in survival_function]
                survival_df = pd.DataFrame(survival_probabilities).transpose()
                survival_df.columns = ['Survival Probability']

                # Plot survival function
                st.line_chart(survival_df)
            except Exception as e:
                st.error(f"Prediction failed: {e}")