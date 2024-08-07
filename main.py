import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from joblib import load
from radiomics import featureextractor
from homee import homee
import SimpleITK as sitk
import tempfile
from scipy.integrate import simps
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import MinMaxScaler
import time

import streamlit.components.v1 as components

# Google Analytics tracking code
tracking_code = """
<script async src="https://www.googletagmanager.com/gtag/js?id=G-66WEQG03ZQ"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-66WEQG03ZQ');
</script>
"""

st.markdown(tracking_code, unsafe_allow_html=True)

# URL des logos hébergés sur GitHub (lien brut)
logo1_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/logo%203.png"
logo2_url = "https://raw.githubusercontent.com/AIxploreRCC/Design/main/images.png"

# Charger le CSS personnalisé
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles2.css")

# Titre de l'application avec logos
st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <div style="display: flex; flex-direction: column; align-items: center; margin-right: 20px;">
            <img src="{logo1_url}" alt="Logo 1" style="width: 120px; height: 90px;">
            <img src="{logo2_url}" alt="Logo 2" style="width: 60px; height: 60px; margin-top: 10px;">
        </div>
        <h1 style="margin: 0;">RCC Clinical Radiomics Algorithm App</h1>
    </div>
    <hr style="border: 1px solid #ccc;">
""", unsafe_allow_html=True)

# Barre de navigation
menu = ["Algorithm App", "Radiomics Score Generator", "About", "Contact"]
choice = st.selectbox("Navigation", menu, key="main_navigation")

def about():
    st.header("About")
    st.write("""
    This application predicts survival using radiomics and clinical data. Adjust the input variables to see how the survival curve changes.
    
    ### Use of the Application
    RCC Clinical Radiomics Algorithm App is not a registered medical device and cannot be used by patients or clinicians for diagnosis, prevention, monitoring, treatment, or alleviation of diseases. The model is made available for academic research and peer review purposes only.
    
    This application was developed as part of a study to predict recurrence after surgery for high-risk kidney cancer. More information about the development of the algorithm can be found in the manuscript.
    
    ### Permitted Purpose(s)
    This license has been provided for the tool, which is a research-use-only tool, intended to facilitate academic and clinical research into clinical risk prediction modeling for non-metastatic kidney cancer after surgery. The license is limited to use by researchers to gain new knowledge or insights or for peer review purposes.
    
    ### Key Features:
    - Upload CT images and segmentation masks to extract radiomic features.
    - Calculate and display radiomic scores.
    - Visualize survival curves based on input variables.
    """)

def contact():
    st.header("Contact")
    st.write("""
    For any inquiries, please contact us at: support@radiomicsapp.com
    """)

# Fonction load_model intégrée
def load_model():
    try:
        return load('random_survival_forest_model.joblib')
    except Exception as e:
        st.error(f"Failed to load the model: {str(e)}")
        raise

rsf_model = load_model()
scaler = load('scaler.joblib')

def setup_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['normalize'] = True
    extractor.settings['normalizeScale'] = 100
    extractor.settings['resampledPixelSpacing'] = [1, 1, 1]
    extractor.settings['interpolator'] = sitk.sitkBSpline
    extractor.settings['binWidth'] = 25
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0, 4.0, 5.0]})
    return extractor

def display_images(ct_image, seg_image, slice_number):
    ct_array = sitk.GetArrayFromImage(ct_image)
    seg_array = sitk.GetArrayFromImage(seg_image)

    # Abdominal window setting
    window_level = 30
    window_width = 300
    min_intensity = window_level - window_width // 2
    max_intensity = window_level + window_width // 2

    # Normalize CT image for display
    ct_array = np.clip(ct_array, min_intensity, max_intensity)
    ct_array = (ct_array - min_intensity) / (max_intensity - min_intensity)

    # Resize the images
    ct_resized = ct_array[slice_number, :, :]
    seg_resized = seg_array[slice_number, :, :]

    plt.figure(figsize=(6, 6))  # Adjust the size as needed
    plt.imshow(ct_resized, cmap='gray')
    plt.imshow(seg_resized, cmap='hot', alpha=0.5)
    plt.axis('off')
    st.pyplot(plt)

if choice == "Algorithm App":
    homee()

elif choice == "Radiomics Score Generator":
    uploaded_ct = st.file_uploader("Upload CT Image", type=["nii", "nii.gz"])
    uploaded_seg = st.file_uploader("Upload Segmentation Mask", type=["nii", "nii.gz"])

    if uploaded_ct and uploaded_seg:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_ct:
            tmp_ct.write(uploaded_ct.getvalue())
            tmp_ct.seek(0)
            ct_image = sitk.ReadImage(tmp_ct.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_seg:
            tmp_seg.write(uploaded_seg.getvalue())
            tmp_seg.seek(0)
            seg_image = sitk.ReadImage(tmp_seg.name)

        slice_number = st.slider('Select Slice', 0, ct_image.GetSize()[2] - 1, ct_image.GetSize()[2] // 2)
        display_images(ct_image, seg_image, slice_number)

    if st.button('Start Feature Extraction'):
      with st.spinner('Feature extraction in progress...'):
        try:
            extractor = setup_extractor()
            feature_extraction_result = extractor.execute(ct_image, seg_image)
            features_df = pd.DataFrame([feature_extraction_result])
            
            features_of_interest = [
                'original_firstorder_10Percentile', 'original_firstorder_Mean', 'original_firstorder_Uniformity', 
                'original_glcm_ClusterTendency', 'original_glcm_Idm', 'original_glcm_Imc2', 'original_glcm_JointEnergy',
                'original_gldm_LargeDependenceEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 
                'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 
                'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 
                'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 
                'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_ZoneVariance', 
                'wavelet-LLH_firstorder_Entropy', 'wavelet-LLH_firstorder_InterquartileRange', 'wavelet-LLH_firstorder_Kurtosis', 
                'wavelet-LLH_glcm_Contrast', 'wavelet-LLH_glcm_DifferenceVariance', 'wavelet-LLH_glcm_Idm', 'wavelet-LLH_glcm_Idn', 
                'wavelet-LLH_glcm_Imc1', 'wavelet-LLH_gldm_HighGrayLevelEmphasis', 'wavelet-LLH_gldm_LargeDependenceEmphasis', 
                'wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LLH_glrlm_GrayLevelNonUniformityNormalized', 
                'wavelet-LLH_glrlm_HighGrayLevelRunEmphasis', 'wavelet-LLH_glrlm_LongRunLowGrayLevelEmphasis', 
                'wavelet-LLH_glrlm_RunLengthNonUniformity', 'wavelet-LLH_glrlm_RunPercentage', 'wavelet-LLH_ngtdm_Busyness', 
                'wavelet-LHL_firstorder_RobustMeanAbsoluteDeviation', 'wavelet-LHL_glcm_ClusterTendency', 'wavelet-LHL_glcm_Correlation', 
                'wavelet-LHL_glcm_DifferenceEntropy', 'wavelet-LHL_glcm_Idmn', 'wavelet-LHL_glcm_JointEntropy', 'wavelet-LHL_glcm_SumAverage', 
                'wavelet-LHL_gldm_DependenceNonUniformityNormalized', 'wavelet-LHL_glrlm_LongRunEmphasis', 'wavelet-LHL_glszm_SizeZoneNonUniformityNormalized', 
                'wavelet-LHL_ngtdm_Complexity', 'wavelet-LHH_firstorder_RootMeanSquared'
            ]

            selected_features_df = features_df[features_of_interest]

            st.session_state['selected_features_df'] = selected_features_df

            st.write("Selected Features:")
            st.dataframe(selected_features_df)
        except Exception as e:
            st.error(f"Error during feature extraction: {str(e)}")

    if 'selected_features_df' in st.session_state and st.button('Calculate RAD-Score for Uploaded Patient'):
        try:
            time_points = np.linspace(0, 60, 61)
            cumulative_hazards = rsf_model.predict_cumulative_hazard_function(st.session_state['selected_features_df'])
            rad_scores = np.array([simps([ch(tp) for tp in time_points], time_points) for ch in cumulative_hazards])
            normalized_rad_scores = scaler.transform(rad_scores.reshape(-1, 1)).flatten()
            st.write(f"Normalized RAD-Score for the uploaded patient: {normalized_rad_scores[0]:.5f}")
        except Exception as e:
            st.error(f"Error during RAD-Score calculation: {str(e)}")

elif choice == "About":
    about()
    
elif choice == "Contact":
    contact()

