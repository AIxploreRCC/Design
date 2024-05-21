import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


# URL des logos hébergés sur GitHub (lien brut)
logo1_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/logo%203.png"
logo2_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/images.png"

# Titre de l'application avec logos
st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center;">
        <img src="{logo1_url}" alt="Logo 1" style="width: 60px; height: 60px; margin-bottom: 10px;">
        <img src="{logo2_url}" alt="Logo 2" style="width: 60px; height: 60px; margin-bottom: 10px;">
        <h1 style="margin: 0; text-align: center;">RenalCheck — RCC Clinical Radiomics Algorithm App</h1>
    </div>
    <hr style="border: 1px solid #ccc;">
""", unsafe_allow_html=True)

# Barre de navigation
menu = ["Home", "About", "Radiomics Score Generator", "Contact"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.header("Input Variables")

    # Entrée des variables par l'utilisateur
    var1 = st.sidebar.slider('Variable 1', 0.0, 1.0, 0.5)
    var2 = st.sidebar.slider('Variable 2', 0.0, 1.0, 0.5)
    var3 = st.sidebar.slider('Variable 3', 0.0, 1.0, 0.5)
    var4 = st.sidebar.slider('Variable 4', 0.0, 1.0, 0.5)

    # Option pour télécharger le scanner
    st.header("Radiomics Score Generator")
    uploaded_file = st.file_uploader("Upload your scan and segmentation", type=["nii", "nii.gz", "dcm"])
    if uploaded_file is not None:
        radiomics_score = np.random.rand()  # Utilisation de generate_radiomics_score
        st.write(f"Generated Radiomics Score: {radiomics_score:.2f}")
    else:
        radiomics_score = st.slider('Radiomics Score', 0.0, 1.0, 0.5)

    # Afficher des courbes de survie factices
    st.header("Kaplan-Meier Survival Curve")
    kmf = KaplanMeierFitter()
    data = pd.DataFrame({
        'time': np.random.exponential(10, 100),
        'event': np.random.binomial(1, 0.5, 100),
        'risk_group': np.random.binomial(1, 0.5, 100)
    })

    for name, grouped_df in data.groupby('risk_group'):
        kmf.fit(grouped_df['time'], event_observed=grouped_df['event'], label=f'Risk Group {name}')
        kmf.plot_survival_function()

    plt.title('Kaplan-Meier Survival Curve')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    st.pyplot()

elif choice == "About":
    st.header("About")
    st.write("""
    This application predicts survival using radiomics and clinical data.
    Adjust the input variables to see how the survival curve changes.
    """)

elif choice == "Radiomics Score Generator":
    radiomics_score_generator()

elif choice == "Contact":
    st.header("Contact")
    st.write("""
    For any inquiries, please contact us at: support@radiomicsapp.com
    """)

# Exécuter l'application
if __name__ == '__main__':
    st.run()
