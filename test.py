import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from radiomics_score_generator import radiomics_score_generator
from home import home  # Assurez-vous que l'importation est correcte

# URL des logos hébergés sur GitHub (lien brut)
logo1_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/logo%203.png"
logo2_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/images.png"

# Titre de l'application avec logos
st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <img src="{logo1_url}" alt="Logo 1" style="width: 100px; height: 100px; margin-right: 20px;">
        <img src="{logo2_url}" alt="Logo 2" style="width: 60px; height: 60px; margin-right: 20px;">
        <h1 style="margin: 0; text-align: center;">RenalCheck — RCC Clinical Radiomics Algorithm App</h1>
    </div>
    <hr style="border: 1px solid #ccc;">
""", unsafe_allow_html=True)

# Barre de navigation
menu = ["Home", "About", "Radiomics Score Generator", "Contact"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    home()

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
