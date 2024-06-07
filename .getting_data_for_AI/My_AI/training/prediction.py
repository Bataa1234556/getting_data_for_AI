import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from model_training import preprocess_sample_data

# Load label_encoders, scaler, poly
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')

# Define or provide sample_data
sample_data = {
    'Uildverlegch': [4],  # List of values for each column
    'Mark': [137],
    'Motor_bagtaamj': [3500],
    'Xrop': [1],
    'Joloo': [1],
    'Uildverlesen_on': [2014],
    'Orj_irsen_on': [2022],
    'Hudulguur': [0],
    'Hutlugch': [2],
    'Yavsan_km': [76000]
}

# Load the trained GBR model
loaded_gbr_model = joblib.load('gbr_models.pkl')



# Preprocess sample data
sample_features_tensor = preprocess_sample_data(sample_data, label_encoders, scaler, poly)

# Make predictions using Gradient Boosting Regressor
gbr_prediction = loaded_gbr_model.predict(sample_features_tensor.numpy())
print("Predicted Price:", gbr_prediction[0])
print("Gradient Boosting Regressor Prediction:", gbr_prediction)
