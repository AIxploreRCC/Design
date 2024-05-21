import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Fonction factice pour générer un score radiomique
def generate_radiomics_score(uploaded_file):
    return np.random.rand()

# Style CSS pour la mise en page
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: white;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton > button {
        background-color: #007BFF;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .gray-box {
        background-color: #f0f0f0;
        border: 1px solid #d3d3d3;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("Radiomics Survival Prediction App")

# Barre de navigation
menu = ["Home", "About", "Upload", "Contact"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.header("Input Variables")

    # Entrée des variables par l'utilisateur dans des boîtes grises
    with st.sidebar:
        st.markdown('<div class="gray-box">', unsafe_allow_html=True)
        var1 = st.slider('Variable 1', 0.0, 1.0, 0.5)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="gray-box">', unsafe_allow_html=True)
        var2 = st.slider('Variable 2', 0.0, 1.0, 0.5)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="gray-box">', unsafe_allow_html=True)
        var3 = st.slider('Variable 3', 0.0, 1.0, 0.5)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="gray-box">', unsafe_allow_html=True)
        var4 = st.slider('Variable 4', 0.0, 1.0, 0.5)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="gray-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your scan and segmentation", type=["nii", "nii.gz", "dcm"])
        if uploaded_file is not None:
            radiomics_score = generate_radiomics_score(uploaded_file)
            st.write(f"Generated Radiomics Score: {radiomics_score:.2f}")
        else:
            radiomics_score = st.slider('Radiomics Score', 0.0, 1.0, 0.5)
        st.markdown('</div>', unsafe_allow_html=True)

        # Boutons Compute et Reset
        if st.button("Compute"):
            st.session_state.compute = True

        if st.button("Reset"):
            st.session_state.compute = False
            st.experimental_rerun()

    if 'compute' in st.session_state and st.session_state.compute:
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

elif choice == "Upload":
    st.header("Upload")
    st.write("""
    Upload your scan and segmentation to generate a radiomics score.
    """)

elif choice == "Contact":
    st.header("Contact")
    st.write("""
    For any inquiries, please contact us at: support@radiomicsapp.com
    """)

# Exécuter l'application
if __name__ == '__main__':
    st.run()
