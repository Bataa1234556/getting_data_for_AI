import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib
import xgboost as xgb
import os 

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
    # Ensure data values are lists
    for key, value in data.items():
        if not isinstance(value, list):
            data[key] = [value]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Encode categorical columns
    categorical_columns = ['Uildverlegch', 'Mark', 'Xrop', 'Joloo', 'Hudulguur', 'Hutlugch']
    for col in categorical_columns:
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        else:
            df[col] = -1

    # Scale numerical columns
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

    # Convert to tensor
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
    y_pred = target_scaler.inverse_transform(y_pred)
    y_true = target_scaler.inverse_transform(y_true)
    diff = np.abs(y_pred - y_true)
    accurate_preds = (diff <= tolerance * np.abs(y_true)).sum()
    return accurate_preds

file_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\My_AI\excel\version1.xlsx'
features_tensor, target_tensor, label_encoders, scaler, target_scaler, poly = preprocess_data(file_path)
print("Data preprocessed successfully!")

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
train_features = features_tensor[train_indices].numpy()
val_features = features_tensor[train_size:].numpy()
train_targets = target_tensor[train_indices].numpy().ravel()
val_targets = target_tensor[train_size:].numpy().ravel()

# Define XGBoost model
xgboost_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# Train XGBoost model
xgboost_model.fit(train_features, train_targets)

# Evaluate XGBoost model
y_pred = xgboost_model.predict(val_features)
y_true = val_targets

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print("MSE: ", mse)
print("MAE: ", mae)
print("R2: ", r2)
print("MAPE: ", mape)

# Save XGBoost model
# joblib.dump(xgboost_model, 'xgboost_model.joblib')
# joblib.dump(poly, 'poly.joblib')
# joblib.dump(label_encoders, 'labelencoder.joblib')
# joblib.dump(scaler, 'scaler.joblib')
# joblib.dump(target_scaler, 'target_scaler.joblib')

# Load XGBoost model
# loaded_xgboost_model = joblib.load('xgboost_model.joblib')

# # Make predictions using loaded XGBoost model
# sample_data = {'Uildverlegch': ['7'], 'Mark': ['76'], 'Xrop': ['1'], 'Joloo': ['0'], 'Hudulguur': ['0'], 'Hutlugch': ['1'], 'Motor_bagtaamj': [1500], 'Uildverlesen_on': [2011], 'Orj_irsen_on': [2021], 'Yavsan_km': [120000]}
# sample_features_tensor = preprocess_sample_data(sample_data, label_encoders, scaler, poly)
# sample_pred = loaded_xgboost_model.predict(sample_features_tensor.numpy())
# print("Predicted price: ", target_scaler.inverse_transform(sample_pred.reshape(-1, 1))[0][0])









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
