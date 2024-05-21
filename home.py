import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import plotly.graph_objects as go
import os


# Function to load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the custom CSS file
local_css("styles.css")


# Charger le modèle avec mise en cache
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'coxph_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load(model_path)

# Charger le modèle
model_cox = load_model()

# Seuil optimal pour séparer les groupes de risque
optimal_threshold = 3.38141178443309

def home():
    st.write("""
    RenalCheck is an advanced AI algorithm designed to predict post-operative oncological outcomes 
    in patients with clear renal cell carcinoma (RCC). This tool is tailored for patients at intermediate or high risk of recurrence, specifically 
    those meeting the eligibility criteria of the KEYNOTE 564 trial, including stages pT1b and G3-4, pT3/pT4, and N1.
    """)

    # Inputs for the model's variables
    hb = st.selectbox("Hemoglobin < lower limit of normal", options=[0, 1])
    N = st.selectbox("N", options=[0, 1, 2])
    rad = st.slider("Radiomics Signature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    Thrombus = st.selectbox("Thrombus", options=[0, 1, 2, 3])

    # DataFrame for model input
    input_df = pd.DataFrame({
        'HbN': [hb],
        'rad': [rad],
        'N': [N],
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
                time_points = time_points[time_points <= 60]  # Stop at 60 months
                survival_probabilities = [fn(time_points) for fn in survival_function]
                survival_df = pd.DataFrame(survival_probabilities).transpose()
                survival_df.columns = ['Survival Probability']

                # Create a table for Disease Free Survival (DFS) Probability at 12-month intervals up to 60 months
                intervals = np.arange(12, 61, 12)
                dfs_probabilities = survival_df.iloc[[int(interval - 1) for interval in intervals if interval <= len(survival_df)]].reset_index(drop=True)
                dfs_probabilities.index = [f"{month} months" for month in intervals]
                dfs_probabilities = dfs_probabilities.T
                st.table(dfs_probabilities)

                # Plot survival function using Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time_points, y=survival_df['Survival Probability'], mode='lines', name='Survival Probability'))
                fig.update_layout(title='Kaplan-Meier Curve', xaxis_title='Time (months)', yaxis_title='Survival Probability')
                st.plotly_chart(fig)

                # Calculate the risk score
                risk_score = model_cox.predict(input_df)[0]
                st.write(f"Calculated risk score: {risk_score:.5f}")

                # Determine the risk group
                risk_group = "High risk" if risk_score >= optimal_threshold else "Low risk"
                st.subheader(f"The patient is in the {risk_group} group.")
                
                # Add patient-specific prediction text
                st.subheader('Patient-specific prediction')
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Run the home function
if __name__ == "__main__":
    home()
