import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from joblib import load
from lifelines import KaplanMeierFitter
import os

# Function to load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the custom CSS file
local_css("styles.css")

# URL des logos hébergés sur GitHub (lien brut)
logo1_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/logo%203.png"
logo2_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/images.png"

# Titre de l'application avec logos
st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="{logo1_url}" alt="Logo 1" style="width: 150px; height: 100px; margin-right: 20px;">
        <img src="{logo2_url}" alt="Logo 2" style="width: 60px; height: 60px; margin-right: 20px;">
        <h1 style="margin: 0; text-align: center;">RenalCheck — RCC Clinical Radiomics Algorithm App</h1>
    </div>
    <hr style="border: 1px solid #ccc;">
""", unsafe_allow_html=True)

# Barre de navigation
menu = ["Home", "About", "Radiomics Score Generator", "Contact"]
choice = st.selectbox("Navigation", menu, key="main_navigation")

def load_model():
    model_path = 'coxph_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load(model_path)

def load_km_data():
    file_path = "/mnt/data/km_curve_data.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df

def plot_kaplan_meier(data):
    kmf = KaplanMeierFitter()
    
    fig = go.Figure()

    for group in data['group'].unique():
        mask = data['group'] == group
        kmf.fit(data[mask]['TimeR'], event_observed=data[mask]['Rec'], label=group)
        survival_function = kmf.survival_function_
        fig.add_trace(go.Scatter(
            x=survival_function.index, 
            y=survival_function.iloc[:, 0],
            mode='lines',
            name=group
        ))

    fig.update_layout(title='Kaplan-Meier Curve',
                      xaxis_title='Time (months)',
                      yaxis_title='Survival Probability')
    
    return fig

def home():
    st.write("""
    RenalCheck is an advanced AI algorithm designed to predict post-operative oncological outcomes 
    in patients with clear renal cell carcinoma (RCC). This tool is tailored for patients at intermediate or high risk of recurrence, specifically 
    those meeting the eligibility criteria of the KEYNOTE 564 trial, including stages pT1b and G3-4, pT3/pT4, and N1.
    """)

    col1, col2 = st.columns(2)

    with col1:
        hb = st.selectbox("Hemoglobin < lower limit of normal", options=[0, 1])
        N = st.selectbox("Pathological Lymph Node Involvement", options=[0, 1, 2])
        rad = st.slider("Radiomics Signature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        Thrombus = st.selectbox("Vascular Invasion", options=[0, 1, 2, 3])

    with col2:
        input_df = pd.DataFrame({
            'HbN': [hb],
            'rad': [rad],
            'N': [N],
            'Thrombus': [Thrombus]
        })

        input_df['N'] = input_df['N'].astype('category')
        input_df['Thrombus'] = input_df['Thrombus'].astype('category')

        if st.button('Predict Survival'):
            with st.spinner('Calculating... Please wait.'):
                try:
                    model_cox = load_model()
                    survival_function = model_cox.predict_survival_function(input_df)
                    st.subheader('Estimated Survival Probability:')
                    
                    time_points = survival_function[0].x
                    time_points = time_points[time_points <= 60]
                    survival_probabilities = [fn(time_points) for fn in survival_function]
                    survival_df = pd.DataFrame(survival_probabilities).transpose()
                    survival_df.columns = ['Survival Probability']

                    intervals = np.arange(12, 61, 12)
                    dfs_probabilities = survival_df.iloc[[int(interval - 1) for interval in intervals if interval <= len(survival_df)]].reset_index(drop=True)
                    dfs_probabilities.index = [f"{month} months" for month in intervals]
                    dfs_probabilities = dfs_probabilities.T
                    st.table(dfs_probabilities)

                    data = load_km_data()
                    fig = plot_kaplan_meier(data)
                    
                    fig.add_trace(go.Scatter(x=time_points, y=survival_df['Survival Probability'], mode='lines', name='Patient-specific prediction', line=dict(color='blue', dash='dot')))
                    fig.update_layout(title='Kaplan-Meier Curve', xaxis_title='Time (months)', yaxis_title='Survival Probability')
                    st.plotly_chart(fig)

                    risk_score = model_cox.predict(input_df)[0]
                    st.write(f"Calculated risk score: {risk_score:.5f}")

                    risk_group = "High risk" if risk_score >= optimal_threshold else "Low risk"
                    st.subheader(f"The patient is in the {risk_group} group.")
                    st.subheader('Patient-specific prediction')

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

def about():
    st.header("About")
    st.write("""
    This application predicts survival using radiomics and clinical data.
    Adjust the input variables to see how the survival curve changes.
    """)

def radiomics_score_generator():
    st.header("Radiomics Score Generator")
    st.write("This feature is under development.")

def contact():
    st.header("Contact")
    st.write("""
    For any inquiries, please contact us at: support@radiomicsapp.com
    """)

if choice == "Home":
    home()
elif choice == "About":
    about()
elif choice == "Radiomics Score Generator":
    radiomics_score_generator()
elif choice == "Contact":
    contact()
