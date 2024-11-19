import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_regression_model.pkl')

# Load the individual encoders
flat_model_encoder = joblib.load('flat_model_encoder.pkl')
flat_type_encoder = joblib.load('flat_type_encoder.pkl')
storey_range_encoder = joblib.load('storey_range_encoder.pkl')
town_encoder = joblib.load('town_encoder.pkl')

# Streamlit UI to input flat details
st.title('Resale Price Prediction for Flats')

# Input fields for the user to enter details about the flat
town = st.selectbox('Select Town:', ['ang_mo_kio', 'bedok', 'bishan', 'bukit_batok', 'bukit_merah', 'bukit_panjang', 'central_area', 'choa_chu_kang', 'clementi', 'geylang', 'hougang', 'jurong_east', 'jurong_west', 'kallang/whampoa', 'lim_chu_kang', 'marine_parade', 'pasir_ris', 'punggol', 'queenstown', 'sembawang', 'sengkang', 'serangoon', 'tampines', 'toa_payoh', 'woodlands', 'yishun'])
flat_type = st.selectbox('Select Flat Type:', ['1_room', '2_room', '3_room', '4_room', '5_room', 'executive', 'multi_generation'])
storey_range = st.selectbox('Select Storey Range:', ['01_to_03', '01_to_05', '04_to_06', '06_to_10', '07_to_09', '10_to_12', '11_to_15', '13_to_15', '16_to_18', '16_to_20', '19_to_21', '21_to_25', '22_to_24', '25_to_27', '26_to_30', '28_to_30', '31_to_33', '31_to_35', '34_to_36', '36_to_40', '37_to_39', '40_to_42', '43_to_45', '46_to_48', '49_to_51'])
flat_model = st.selectbox('Select Flat Model:', ['2-room', '3gen', 'adjoined_flat', 'apartment', 'dbss', 'improved', 'improved-maisonette', 'maisonette', 'model_a', 'model_a-maisonette', 'model_a2', 'multi_generation', 'new_generation', 'premium_apartment', 'premium_apartment_loft', 'premium_maisonette', 'simplified', 'standard', 'terrace', 'type_s1', 'type_s2'])
floor_area_sqm = st.number_input('Enter Floor Area (sqm):', min_value=20, max_value=300, value=80)
age_of_flat = st.number_input('Enter Age of Flat (years):', min_value=0, max_value=50, value=10)
remaining_lease_months = st.number_input('Enter Remaining Lease (months):', min_value=1, max_value=999, value=120)

# Prepare the input data as a DataFrame
input_data = {
    'floor_area_sqm': floor_area_sqm,
    'age_of_flat': age_of_flat,
    'remaining_lease_months': remaining_lease_months,
    'flat_type_1_room': 1 if flat_type == '1_room' else 0,
    'flat_type_2_room': 1 if flat_type == '2_room' else 0,
    'flat_type_3_room': 1 if flat_type == '3_room' else 0,
    'flat_type_4_room': 1 if flat_type == '4_room' else 0,
    'flat_type_5_room': 1 if flat_type == '5_room' else 0,
    'flat_type_executive': 1 if flat_type == 'executive' else 0,
    'flat_type_multi_generation': 1 if flat_type == 'multi_generation' else 0,

    'town_ang_mo_kio': 1 if town == 'ang_mo_kio' else 0,
    'town_bedok': 1 if town == 'bedok' else 0,
    'town_bishan': 1 if town == 'bishan' else 0,
    'town_bukit_batok': 1 if town == 'bukit_batok' else 0,
    'town_bukit_merah': 1 if town == 'bukit_merah' else 0,
    'town_bukit_panjang': 1 if town == 'bukit_panjang' else 0,
    'town_bukit_timah': 1 if town == 'bukit_timah' else 0,
    'town_central_area': 1 if town == 'central_area' else 0,
    'town_choa_chu_kang': 1 if town == 'choa_chu_kang' else 0,
    'town_clementi': 1 if town == 'clementi' else 0,
    'town_geylang': 1 if town == 'geylang' else 0,
    'town_hougang': 1 if town == 'hougang' else 0,
    'town_jurong_east': 1 if town == 'jurong_east' else 0,
    'town_jurong_west': 1 if town == 'jurong_west' else 0,
    'town_kallang/whampoa': 1 if town == 'kallang/whampoa' else 0,
    'town_lim_chu_kang': 1 if town == 'lim_chu_kang' else 0,
    'town_marine_parade': 1 if town == 'marine_parade' else 0,
    'town_pasir_ris': 1 if town == 'pasir_ris' else 0,
    'town_punggol': 1 if town == 'punggol' else 0,
    'town_queenstown': 1 if town == 'queenstown' else 0,
    'town_sembawang': 1 if town == 'sembawang' else 0,
    'town_sengkang': 1 if town == 'sengkang' else 0,
    'town_serangoon': 1 if town == 'serangoon' else 0,
    'town_tampines': 1 if town == 'tampines' else 0,
    'town_toa_payoh': 1 if town == 'toa_payoh' else 0,
    'town_woodlands': 1 if town == 'woodlands' else 0,
    'town_yishun': 1 if town == 'yishun' else 0,

    'storey_range_01_to_03': 1 if storey_range == '01_to_03' else 0,
    'storey_range_01_to_05': 1 if storey_range == '01_to_05' else 0,
    'storey_range_04_to_06': 1 if storey_range == '04_to_06' else 0,
    'storey_range_06_to_10': 1 if storey_range == '06_to_10' else 0,
    'storey_range_07_to_09': 1 if storey_range == '07_to_09' else 0,
    'storey_range_10_to_12': 1 if storey_range == '10_to_12' else 0,
    'storey_range_11_to_15': 1 if storey_range == '11_to_15' else 0,
    'storey_range_13_to_15': 1 if storey_range == '13_to_15' else 0,
    'storey_range_16_to_18': 1 if storey_range == '16_to_18' else 0,
    'storey_range_16_to_20': 1 if storey_range == '16_to_20' else 0,
    'storey_range_19_to_21': 1 if storey_range == '19_to_21' else 0,
    'storey_range_21_to_25': 1 if storey_range == '21_to_25' else 0,
    'storey_range_22_to_24': 1 if storey_range == '22_to_24' else 0,
    'storey_range_25_to_27': 1 if storey_range == '25_to_27' else 0,
    'storey_range_26_to_30': 1 if storey_range == '26_to_30' else 0,
    'storey_range_28_to_30': 1 if storey_range == '28_to_30' else 0,
    'storey_range_31_to_33': 1 if storey_range == '31_to_33' else 0,
    'storey_range_31_to_35': 1 if storey_range == '31_to_35' else 0,
    'storey_range_34_to_36': 1 if storey_range == '34_to_36' else 0,
    'storey_range_36_to_40': 1 if storey_range == '36_to_40' else 0,
    'storey_range_37_to_39': 1 if storey_range == '37_to_39' else 0,
    'storey_range_40_to_42': 1 if storey_range == '40_to_42' else 0,
    'storey_range_43_to_45': 1 if storey_range == '43_to_45' else 0,
    'storey_range_46_to_48': 1 if storey_range == '46_to_48' else 0,
    'storey_range_49_to_51': 1 if storey_range == '49_to_51' else 0,

    'flat_model_2-room': 1 if flat_model == '2-room' else 0,
    'flat_model_3gen': 1 if flat_model == '3gen' else 0,
    'flat_model_adjoined_flat': 1 if flat_model == 'adjoined_flat' else 0,
    'flat_model_apartment': 1 if flat_model == 'apartment' else 0,
    'flat_model_dbss': 1 if flat_model == 'dbss' else 0,
    'flat_model_improved': 1 if flat_model == 'improved' else 0,
    'flat_model_improved-maisonette': 1 if flat_model == 'improved-maisonette' else 0,
    'flat_model_maisonette': 1 if flat_model == 'maisonette' else 0,
    'flat_model_model_a': 1 if flat_model == 'model_a' else 0,
    'flat_model_model_a-maisonette': 1 if flat_model == 'model_a-maisonette' else 0,
    'flat_model_model_a2': 1 if flat_model == 'model_a2' else 0,
    'flat_model_multi_generation': 1 if flat_model == 'multi_generation' else 0,
    'flat_model_new_generation': 1 if flat_model == 'new_generation' else 0,
    'flat_model_premium_apartment': 1 if flat_model == 'premium_apartment' else 0,
    'flat_model_premium_apartment_loft': 1 if flat_model == 'premium_apartment_loft' else 0,
    'flat_model_premium_maisonette': 1 if flat_model == 'premium_maisonette' else 0,
    'flat_model_simplified': 1 if flat_model == 'simplified' else 0,
    'flat_model_standard': 1 if flat_model == 'standard' else 0,
    'flat_model_terrace': 1 if flat_model == 'terrace' else 0,
    'flat_model_type_s1': 1 if flat_model == 'type_s1' else 0,
    'flat_model_type_s2': 1 if flat_model == 'type_s2' else 0,

}

input_df = pd.DataFrame([input_data])

if st.button('Predict Resale Price'):
    prediction = model.predict(input_df)
    st.write(f'Predicted Resale Price: SGD {prediction[0]:,.2f}')
