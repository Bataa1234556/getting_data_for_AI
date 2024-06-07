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
from sklearn.model_selection import GridSearchCV
import joblib

# Function to preprocess data from an Excel file
def preprocess_data(file_path, degree=2):
    df = pd.read_excel(file_path)
    df.dropna(inplace=True)
    df = df[df['Une'] > 0]

    label_encoders = {}
    categorical_columns = ['Uildverlegch', 'Mark', 'Xrop', 'Joloo', 'Hudulguur', 'Hutlugch']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    numerical_columns = ['Motor_bagtaamj', 'Uildverlesen_on', 'Orj_irsen_on', 'Yavsan_km']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    target_scaler = StandardScaler()
    df['Une'] = target_scaler.fit_transform(df[['Une']])

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[numerical_columns])
    poly_feature_names = poly.get_feature_names_out(numerical_columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    df = df.drop(columns=numerical_columns).join(poly_df)

    features = df[categorical_columns + list(poly_feature_names)]
    target = df['Une']

    # Check for NaNs in features and target
    if features.isnull().any().any():
        raise ValueError("NaNs found in features")
    if target.isnull().any():
        raise ValueError("NaNs found in target")

    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    target_tensor = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)

    return features_tensor, target_tensor, label_encoders, scaler, target_scaler, poly

# Function to preprocess sample data from a dictionary
def preprocess_sample_data(data, label_encoders, scaler, poly):
    df = pd.DataFrame(data)

    categorical_columns = ['Uildverlegch', 'Mark', 'Xrop', 'Joloo', 'Hudulguur', 'Hutlugch']
    for col in categorical_columns:
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        else:
            df[col] = -1

    numerical_columns = ['Motor_bagtaamj', 'Uildverlesen_on', 'Orj_irsen_on', 'Yavsan_km']
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Apply polynomial features
    poly_features = poly.transform(df[numerical_columns])
    poly_feature_names = poly.get_feature_names_out(numerical_columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    df = df.drop(columns=numerical_columns).join(poly_df)

    # Check for NaNs in sample features
    if df.isnull().any().any():
        raise ValueError("NaNs found in sample features")

    features_tensor = torch.tensor(df.values, dtype=torch.float32)

    return features_tensor

# Main script
class CarPriceDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Function to calculate regression accuracy
def regression_accuracy(y_pred, y_true, tolerance=0.1):
    y_pred = target_scaler.inverse_transform(y_pred.detach().numpy())
    y_true = target_scaler.inverse_transform(y_true.detach().numpy())
    diff = np.abs(y_pred - y_true)
    accurate_preds = (diff <= tolerance * np.abs(y_true)).sum()
    return accurate_preds

file_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\My_AI\excel\version1.xlsx'
features_tensor, target_tensor, label_encoders, scaler, target_scaler, poly = preprocess_data(file_path)
print("Datapreprocessed successfully!")

# Split data into training and validation sets
print("Length of features tensor:", len(features_tensor))
print("Length of target tensor:", len(target_tensor))

# Calculate split sizes
# Split indices

train_size = int(0.8 * len(features_tensor))
test_size = len(features_tensor) - train_size
indices = list(range(len(features_tensor)))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create subsets
train_features = features_tensor[train_indices]
val_features = features_tensor[val_indices]
train_targets = target_tensor[train_indices]
val_targets = target_tensor[val_indices]



# Create data loaders
batch_size = 32
train_dataset = CarPriceDataset(train_features, train_targets)
val_dataset = CarPriceDataset(val_features, val_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define Gradient Boosting Regressor model
param_grid = {  
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [3, 4, 5]
}

gb_regressor = GradientBoostingRegressor()
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=param_grid, cv=5)
grid_search.fit(features_tensor.numpy(), target_tensor.numpy().ravel())

print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

# Train Gradient Boosting Regressor model
gb_regressor = GradientBoostingRegressor(**grid_search.best_params_)
gb_regressor.fit(features_tensor.numpy(), target_tensor.numpy().ravel())

# Evaluate Gradient Boosting Regressor model
y_pred = gb_regressor.predict(features_tensor.numpy())
mse = mean_squared_error(target_tensor.numpy().ravel(), y_pred)
mae = mean_absolute_error(target_tensor.numpy().ravel(), y_pred)
r2 = r2_score(target_tensor.numpy().ravel(), y_pred)
print("MSE: ", mse)
print("MAE: ", mae)
print("R2: ", r2)

# Save Gradient Boosting Regressor model
joblib.dump(gb_regressor, 'gb_regressor.joblib')

# Load Gradient Boosting Regressor model
loaded_gb_regressor = joblib.load('gb_regressor.joblib')

# Make predictions using loaded Gradient Boosting Regressor model
sample_data = {'Uildverlegch': ['7'], 'Mark': ['76'], 'Xrop': ['1'], 'Joloo': ['0'], 'Hudulguur': ['0'], 'Hutlugch': ['1'], 'Motor_bagtaamj': [1500], 'Uildverlesen_on': [2006], 'Orj_irsen_on': [2012], 'Yavsan_km': [0]}
sample_features_tensor = preprocess_sample_data(sample_data, label_encoders, scaler, poly)
sample_pred = loaded_gb_regressor.predict(sample_features_tensor.numpy())
print("Predicted price: ", target_scaler.inverse_transform(sample_pred.reshape(-1, 1))[0][0])








# data = {
#     "Manufacturer": ["Honda", "Honda", "Honda", "Honda", "Honda", "Honda"],
#     "Model": ["CR-V", "Fit", "Fit", "CR-Z", "HR-V", "Insight"],
#     "Engine Capacity": [1.5, 1.5, 1.5, 1.5, 1.8, 1.3],
#     "Transmission": ["1", "1", "1", "1", "1", "1"],
#     "Steering": ["Буруу", "Буруу", "Буруу", "Буруу", "Буруу", "Буруу"],
#     "Type": ["Жийп", "Суудлын тэрэг", "Гэр бүлийн", "Суудлын тэрэг", "Гэр бүлийн", "Суудлын тэрэг"],
#     "Color": ["Цагаан", "Мөнгөлөг", "Хөх", "Мөнгөлөг", "Мөнгөлөг", "Саарал"],
#     "Drive": ["Хайбрид", "Хайбрид", "Хайбрид", "Хайбрид", "Бензин", "Бензин"],
#     "Interior Color": ["Хар", "Саарал", "Хар", "Саарал", "Саарал", "Бусад"],
#     "Drive Type": ["Бүх дугуй 4WD", "Бүх дугуй 4WD", "Урдаа FWD", "Урдаа FWD", "Бүх дугуй 4WD", "Урдаа FWD"],
#     "Manufactured Year": [2015, 2016, 2017, 2018, 2019, 2020],
#     "Import Year": [2016, 2017, 2018, 2019, 2020, 2021],
#     "Mileage": [56000, 800000, 100000, 153000, 180, 256000]
# }

# data_tensor = preprocess_sample_data(data, label_encoders, scaler)

# model.eval()
# with torch.no_grad():
#     predicted_price_tensor = model(data_tensor)
#     predicted_prices = predicted_price_tensor.numpy()

# predicted_prices_actual = target_scaler.inverse_transform(predicted_prices)
# predicted_prices_mnt = [int(price[0] * 1000000) for price in predicted_prices_actual]
# print("Predicted prices (in MNT):", predicted_prices_mnt)
