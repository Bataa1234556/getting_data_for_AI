import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

def preprocess_data(file_path):
    df = pd.read_excel(file_path)

    df.dropna(inplace=True)

    label_encoders = {}
    categorical_columns = ['Manufacturer', 'Model', 'Transmission', 'Steering', 'Type', 'Color', 'Drive', 'Interior Color', 'Drive Type']

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    numerical_columns = ['Engine Capacity', 'Manufactured Year', 'Import Year', 'Mileage']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    features = df[categorical_columns + numerical_columns]
    target = df['Price']

    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    target_tensor = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)

    return features_tensor, target_tensor

# Correct file path
file_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\My_AI\excel\car_data (1).xlsx'

# Check if the directory exists
dir_path = os.path.dirname(file_path)
if os.path.exists(dir_path):
    print(f"Directory exists: {dir_path}")
    print("Files in directory:", os.listdir(dir_path))
else:
    print(f"Directory not found: {dir_path}")

# Check if the file exists
if os.path.exists(file_path):
    print(f"File exists: {file_path}")
    features_tensor, target_tensor = preprocess_data(file_path)
    print("Data preprocessing completed successfully")
else:
    print(f"File not found: {file_path}")
