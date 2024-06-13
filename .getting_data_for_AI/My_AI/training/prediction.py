import joblib
import torch
from model_training import preprocess_sample_data

# Load the Gaussian Process Regressor model
loaded_gp_regressor = joblib.load('gp_regressor.joblib')

# Load preprocessing components
label_encoders = joblib.load('label_encoders.joblib')
scaler = joblib.load('scaler.joblib')
target_scaler = joblib.load('target_scaler.joblib')
poly = joblib.load('poly.joblib')

# Sample data for prediction
sample_data = {
    'Uildverlegch': ['7'],
    'Mark': ['165'],
    'Xrop': ['0'],
    'Joloo': ['1'],
    'Hudulguur': ['0'],
    'Hutlugch': ['1'],
    'Motor_bagtaamj': [10000],
    'Uildverlesen_on': [2010],
    'Orj_irsen_on': [2020],
    'Yavsan_km': [900000]
}

# Preprocess the sample data
sample_features_tensor = preprocess_sample_data(sample_data, label_encoders, scaler, poly)

# Make prediction using the loaded model
sample_pred, sample_std = loaded_gp_regressor.predict(sample_features_tensor.numpy(), return_std=True)

# Inverse transform the prediction to get the actual price
predicted_price = target_scaler.inverse_transform(sample_pred.reshape(-1, 1))[0][0]

print("Predicted price: ", predicted_price)

