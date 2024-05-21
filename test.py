import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Fonction pour générer des données de survie factices
def generate_data(n=100):
    np.random.seed(42)
    data = pd.DataFrame({
        'time': np.random.exponential(10, n),
        'event': np.random.binomial(1, 0.5, n),
        'variable1': np.random.normal(0, 1, n),
        'variable2': np.random.normal(0, 1, n),
        'variable3': np.random.normal(0, 1, n),
        'variable4': np.random.normal(0, 1, n)
    })
    return data

# Charger les données
data = generate_data()

# Titre de l'application
st.title("Survival Analysis App")

# Barre de navigation
menu = ["Home", "About", "Contact"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.header("Input Variables")

    # Entrée des variables par l'utilisateur
    var1 = st.sidebar.slider('Variable 1', float(data['variable1'].min()), float(data['variable1'].max()), float(data['variable1'].mean()))
    var2 = st.sidebar.slider('Variable 2', float(data['variable2'].min()), float(data['variable2'].max()), float(data['variable2'].mean()))
    var3 = st.sidebar.slider('Variable 3', float(data['variable3'].min()), float(data['variable3'].max()), float(data['variable3'].mean()))
    var4 = st.sidebar.slider('Variable 4', float(data['variable4'].min()), float(data['variable4'].max()), float(data['variable4'].mean()))

    # Filtrer les données en fonction des variables
    filtered_data = data[
        (data['variable1'] >= var1) &
        (data['variable2'] >= var2) &
        (data['variable3'] >= var3) &
        (data['variable4'] >= var4)
    ]

    if len(filtered_data) > 0:
        # Kaplan-Meier Fitter
        kmf = KaplanMeierFitter()
        kmf.fit(filtered_data['time'], event_observed=filtered_data['event'])

        # Afficher la courbe de survie
        st.header("Kaplan-Meier Survival Curve")
        fig, ax = plt.subplots()
        kmf.plot_survival_function(ax=ax)
        plt.title('Kaplan-Meier Survival Curve')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        st.pyplot(fig)

        # Interprétation des résultats
        st.subheader("Interpretation")
        st.write("""
        The Kaplan-Meier survival curve shows the estimated probability of survival over time.
        The x-axis represents the time, and the y-axis represents the survival probability.
        """)

    else:
        st.warning("No data available for the selected variables.")

elif choice == "About":
    st.header("About")
    st.write("""
    This application provides survival analysis using the Kaplan-Meier method.
    Adjust the input variables to filter the data and see how the survival curve changes.
    """)

elif choice == "Contact":
    st.header("Contact")
    st.write("""
    For any inquiries, please contact us at: support@survivalanalysisapp.com
    """)

# Exécuter l'application
if __name__ == '__main__':
    st.run()
