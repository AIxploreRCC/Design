import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from joblib import load
from lifelines import KaplanMeierFitter
import os

# Charger le modèle avec mise en cache
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'coxph_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load(model_path)

# Seuil optimal pour séparer les groupes de risque
optimal_threshold = 3.38141178443309

# Charger les données pour tracer la courbe de Kaplan-Meier
@st.cache
def load_km_data():
    file_path = "km_curve_data.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df

# Fonction pour tracer les courbes de Kaplan-Meier
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
    fig.update_layout(
                      xaxis_title='Time (months)',
                      yaxis_title='Survival Probability',
                      width=500,  # Réduire la largeur de la figure
                      height=300)  # Réduire la hauteur de la figure
    
    return fig



def homee():
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

        # Bouton Predict Survival
        predict_button = st.button('Predict Survival')




        input_df = pd.DataFrame({
            'HbN': [hb],
            'rad': [rad],
            'N': [N],
            'Thrombus': [Thrombus]
        })


        
        input_df['N'] = input_df['N'].astype('category')
        input_df['Thrombus'] = input_df['Thrombus'].astype('category')

    with col2:

        if predict_button:
            with st.spinner('Calculating... Please wait.'):
                try:
                    model_cox = load_model()
                    survival_function = model_cox.predict_survival_function(input_df)
                    time_points = survival_function[0].x
                    time_points = time_points[time_points <= 60]
                    survival_probabilities = [fn(time_points) for fn in survival_function]
                    survival_df = pd.DataFrame(survival_probabilities).transpose()
                    survival_df.columns = ['Survival Probability']
                    data = load_km_data()
                    fig = plot_kaplan_meier(data)
                    
                    fig.add_trace(go.Scatter(x=time_points, y=survival_df['Survival Probability'], mode='lines', name='Patient-specific prediction', line=dict(color='blue', dash='dot')))
                    fig.update_layout(xaxis_title='Time (months)', yaxis_title='Survival Probability')
                    st.plotly_chart(fig)

                    risk_score = model_cox.predict(input_df)[0]
                   
                    # Déterminer le groupe de risque et ajouter l'icône de feu appropriée
                    risk_group = "High risk" if risk_score >= optimal_threshold else "Low risk"
                    risk_color = "green" if risk_score >= optimal_threshold else "green"
                    icon_html = f"<div style='text-align: center;'><span style='color: {risk_color}; font-size: 50px;'>&#x1F6A9;</span></div>"
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(icon_html, unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>The patient is in the {risk_group} group.</h3>", unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)


                except Exception as e:
                    st.error(f"Prediction failed: {e}")
