import streamlit as st
import SimpleITK as sitk
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import simps
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from sksurv.ensemble import RandomSurvivalForest
from radiomics import featureextractor

# Liste des caractéristiques d'intérêt
features_of_interest = [
    'original_firstorder_10Percentile', 'original_firstorder_Mean', 
    'original_firstorder_Uniformity', 'original_glcm_ClusterTendency', 
    'original_glcm_Idm', 'original_glcm_Imc2', 'original_glcm_JointEnergy', 
    'original_gldm_LargeDependenceEmphasis', 
    'original_gldm_SmallDependenceLowGrayLevelEmphasis', 
    'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 
    'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_RunVariance', 
    'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 
    'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelVariance', 
    'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 
    'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_ZoneVariance', 
    'wavelet-LLH_firstorder_Entropy', 'wavelet-LLH_firstorder_InterquartileRange', 
    'wavelet-LLH_firstorder_Kurtosis', 'wavelet-LLH_glcm_Contrast', 
    'wavelet-LLH_glcm_DifferenceVariance', 'wavelet-LLH_glcm_Idm', 
    'wavelet-LLH_glcm_Idn', 'wavelet-LLH_glcm_Imc1', 'wavelet-LLH_gldm_HighGrayLevelEmphasis', 
    'wavelet-LLH_gldm_LargeDependenceEmphasis', 'wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis', 
    'wavelet-LLH_glrlm_GrayLevelNonUniformityNormalized', 'wavelet-LLH_glrlm_HighGrayLevelRunEmphasis', 
    'wavelet-LLH_glrlm_LongRunLowGrayLevelEmphasis', 'wavelet-LLH_glrlm_RunLengthNonUniformity', 
    'wavelet-LLH_glrlm_RunPercentage', 'wavelet-LLH_ngtdm_Busyness', 
    'wavelet-LHL_firstorder_RobustMeanAbsoluteDeviation', 'wavelet-LHL_glcm_ClusterTendency', 
    'wavelet-LHL_glcm_Correlation', 'wavelet-LHL_glcm_DifferenceEntropy', 
    'wavelet-LHL_glcm_Idmn', 'wavelet-LHL_glcm_JointEntropy', 'wavelet-LHL_glcm_SumAverage', 
    'wavelet-LHL_gldm_DependenceNonUniformityNormalized', 'wavelet-LHL_glrlm_LongRunEmphasis', 
    'wavelet-LHL_glszm_SizeZoneNonUniformityNormalized', 'wavelet-LHL_ngtdm_Complexity', 
    'wavelet-LHH_firstorder_RootMeanSquared'
]

columns_to_remove = [
    'CT Image Path', 'Segmentation File Path', 'diagnostics_Versions_PyRadiomics', 
    'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK', 
    'diagnostics_Versions_PyWavelet', 'diagnostics_Versions_Python', 
    'diagnostics_Configuration_Settings', 'diagnostics_Configuration_EnabledImageTypes', 
    'diagnostics_Image-original_Hash', 'diagnostics_Image-original_Dimensionality', 
    'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 
    'diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 
    'diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_Hash', 
    'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Size', 
    'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_VoxelNum', 
    'diagnostics_Mask-original_VolumeNum', 'diagnostics_Mask-original_CenterOfMassIndex', 
    'diagnostics_Mask-original_CenterOfMass', 'diagnostics_Image-interpolated_Spacing', 
    'diagnostics_Image-interpolated_Size', 'diagnostics_Image-interpolated_Mean', 
    'diagnostics_Image-interpolated_Minimum', 'diagnostics_Image-interpolated_Maximum', 
    'diagnostics_Mask-interpolated_Spacing', 'diagnostics_Mask-interpolated_Size', 
    'diagnostics_Mask-interpolated_BoundingBox', 'diagnostics_Mask-interpolated_VoxelNum', 
    'diagnostics_Mask-interpolated_VolumeNum', 'diagnostics_Mask-interpolated_CenterOfMassIndex', 
    'diagnostics_Mask-interpolated_CenterOfMass', 'diagnostics_Mask-interpolated_Mean', 
    'diagnostics_Mask-interpolated_Minimum', 'diagnostics_Mask-interpolated_Maximum', 
    'Number of Features'
]

def load_model():
    try:
        return load('/mnt/data/random_survival_forest_model.joblib')
    except Exception as e:
        st.error(f"Failed to load the model: {str(e)}")
        raise

rsf_model = load_model()
scaler = load('/mnt/data/scaler.joblib')

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
    plt.figure()
    plt.imshow(ct_array[slice_number], cmap='gray')
    plt.imshow(seg_array[slice_number], cmap='jet', alpha=0.5)
    plt.axis('off')
    st.pyplot(plt)

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
    progress_bar = st.progress(0)
    extractor = setup_extractor()
    feature_extraction_result = extractor.execute(ct_image, seg_image)
    features_df = pd.DataFrame([feature_extraction_result])

    progress_bar.progress(50)

    # Further processing
    features_df.drop(columns=columns_to_remove, inplace=True)
    selected_features_df = features_df[features_of_interest]

    progress_bar.progress(100)
    st.write("Selected Features:", st.dataframe(selected_features_df))

if 'selected_features_df' in st.session_state and st.button('Calculate RAD-Score for Uploaded Patient'):
    time_points = np.linspace(0, 60, 61)
    cumulative_hazards = rsf_model.predict_cumulative_hazard_function(st.session_state['selected_features_df'])
    rad_scores = np.array([simps([ch(tp) for tp in time_points], time_points) for ch in cumulative_hazards])
    normalized_rad_scores = scaler.transform(rad_scores.reshape(-1, 1)).flatten()
    st.write(f"Normalized RAD-Score for the uploaded patient: {normalized_rad_scores[0]:.5f}")