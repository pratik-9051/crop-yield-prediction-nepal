import streamlit as st
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np

# Function to make predictions
def predict_yield(df_new, model_path="xgb_model.json", scaler_path="scaler.pkl", columns_path="feature_columns.pkl"):
    # Load model, scaler, and column list
    model = xgb.Booster()
    model.load_model(model_path)
    
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(columns_path)
    
    # One-hot encode categorical columns
    df_new_encoded = pd.get_dummies(df_new, columns=['DISTRICT', 'Item'])
    
    # Add missing columns
    for col in feature_columns:
        if col not in df_new_encoded:
            df_new_encoded[col] = 0
    
    # Ensure correct column order
    X_new = df_new_encoded[feature_columns]
    
    # Scale
    X_new_scaled = scaler.transform(X_new)
    
    # Predict
    dnew = xgb.DMatrix(X_new_scaled)
    y_pred_log = model.predict(dnew)
    y_pred = np.expm1(y_pred_log)  # inverse of log1p

    return y_pred

# Streamlit UI
st.title("Crop Yield Prediction")
st.write("Enter the details to predict the crop yield (in kg/ha):")

# Input fields for user to enter data
year = st.number_input("Year", min_value=1900, max_value=2100, value=1982)

precipitation = st.number_input("Precipitation Total (PRECTOT)", value=64.1083)
if precipitation < 0 or precipitation > 1000:
    st.warning("⚠️ Precipitation should be between 0 and 1000 mm")

temperature_mean = st.number_input("Mean Temperature (T2M)", value=18.964)
if temperature_mean < -10 or temperature_mean > 45:
    st.warning("⚠️ Mean temperature should be between -10°C and 45°C")

temperature_max = st.number_input("Max Temperature (T2M_MAX)", value=37.19)
if temperature_max < -10 or temperature_max > 50:
    st.warning("⚠️ Max temperature should be between -10°C and 50°C")

temperature_min = st.number_input("Min Temperature (T2M_MIN)", value=0.44)
if temperature_min < -20 or temperature_min > 40:
    st.warning("⚠️ Min temperature should be between -20°C and 40°C")

humidity = st.number_input("Relative Humidity (RH2M)", value=52.645)
if humidity < 0 or humidity > 100:
    st.warning("⚠️ Humidity should be between 0% and 100%")

district = st.selectbox("District", ['Achham', 'Arghakhanchi', 'Baglung', 'Baitadi', 'Bajang', 'Bajura',
       'Banke', 'Bara', 'Bardiya', 'Bhaktapur', 'Bhojpur', 'Chitawan',
       'Dadeldhura', 'Dailekh', 'Dang', 'Darchula', 'Dhading', 'Dhankuta',
       'Dhanusa', 'Dolkha', 'Dolpa', 'Doti', 'Gorkha', 'Gulmi', 'Humla',
       'Ilam', 'Jajarkot', 'Jhapa', 'Jumla', 'Kabhre', 'Kailali',
       'Kalikot', 'Kanchanpur', 'Kapilbastu', 'Kaski', 'Kathmandu',
       'Khotang', 'Lalitpur', 'Lamjung', 'Mahottari', 'Makwanpur',
       'Manang', 'Morang', 'Mugu', 'Mustang', 'Myagdi', 'Nawalparasi',
       'Nuwakot', 'Okhaldhunga', 'Palpa', 'Panchther', 'Parbat', 'Parsa',
       'Pyuthan', 'Ramechaap', 'Rasuwa', 'Rolpa', 'Routahat', 'Rukum',
       'Rupandehi', 'Salyan', 'Sankhuwasabha', 'Saptari', 'Sarlahi',
       'Sindhuli', 'Sindhupalchowk', 'Siraha', 'Solukhumbu', 'Sunsari',
       'Surkhet', 'Syangja', 'Tanahun', 'Taplejung', 'Terhathum',
       'Udayapur'])  # Add your districts here
item = st.selectbox("Item", ['Paddy', 'Maize', 'Wheat', 'Millet'])  # Add your items here

# Create DataFrame for the new input
data = {
    'YEAR': [year],
    'PRECTOT': [precipitation],
    'T2M': [temperature_mean],
    'T2M_MAX': [temperature_max],
    'T2M_MIN': [temperature_min],
    'RH2M': [humidity],
    'DISTRICT': [district],
    'Item': [item]
}
df_new = pd.DataFrame(data)

# Button to make prediction
if st.button("Predict Yield"):
    # Call the prediction function
    predicted_yield = predict_yield(df_new)
    
    # Display the result
    st.write(f"Predicted Yield: {predicted_yield[0]:.2f} kg/ha")