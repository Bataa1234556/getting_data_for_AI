import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# Function to preprocess data from an Excel file
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

# Function to preprocess sample data from a dictionary
def preprocess_sample_data(data):
    # Create DataFrame from sample data
    df = pd.DataFrame(data)
    
    # Apply LabelEncoder to categorical columns
    categorical_columns = ['Manufacturer', 'Model', 'Transmission', 'Steering', 'Type', 'Color', 'Drive', 'Interior Color', 'Drive Type']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Scale numerical columns
    numerical_columns = ['Engine Capacity', 'Manufactured Year', 'Import Year', 'Mileage']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Convert DataFrame to tensor
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

class CarPriceModel(nn.Module):
    def __init__(self, input_size):
        super(CarPriceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Update this path to your actual file path
file_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\My_AI\excel\car_data (1).xlsx'

# Check if the file exists before proceeding
if os.path.exists(file_path):
    features_tensor, target_tensor = preprocess_data(file_path)
    print("Data preprocessing completed successfully")
else:
    raise FileNotFoundError(f"File not found: {file_path}")

# Check for NaNs and Infs
def check_for_invalid_values(tensor):
    if torch.isnan(tensor).any():
        print("NaNs found in tensor")
    if torch.isinf(tensor).any():
        print("Infs found in tensor")

check_for_invalid_values(features_tensor)
check_for_invalid_values(target_tensor)

# Create dataset and dataloaders
dataset = CarPriceDataset(features_tensor, target_tensor)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
input_size = features_tensor.shape[1]
model = CarPriceModel(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for features, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation on test set
model.eval()
with torch.no_grad():
    test_loss = 0
    for features, targets in test_loader:
        outputs = model(features)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
    test_loss /= len(test_loader)
    
    print(f'Test Loss: {test_loss}')

# Sample data for prediction
data = {
    "Manufacturer": ["Honda", "Honda", "Honda", "Honda", "Honda", "Honda"],
    "Model": ["CR-V", "Fit", "Fit", "CR-Z", "HR-V", "Insight"],
    "Engine Capacity": [1.5, 1.5, 1.5, 1.5, 1.8, 1.3],
    "Transmission": ["Автомат", "Автомат", "Автомат", "Автомат", "Автомат", "Автомат"],
    "Steering": ["Буруу", "Буруу", "Буруу", "Буруу", "Буруу", "Буруу"],
    "Type": ["Жийп", "Суудлын тэрэг", "Гэр бүлийн", "Суудлын тэрэг", "Гэр бүлийн", "Суудлын тэрэг"],
    "Color": ["Хар", "Саарал", "Саарал", "Цэнхэр", "Саарал", "Цагаан"],
    "Manufactured Year": [2021, 2014, 2014, 2011, 2008, 2009],
    "Import Year": [2024, 2020, 2021, 2023, 2018, 2016],
    "Drive": ["Бензин", "Хайбрид", "Хайбрид", "Хайбрид", "Бензин", "Бензин"],
    "Interior Color": ["Хар", "Саарал", "Хар", "Саарал", "Саарал", "Бусад"],
    "Drive Type": ["Бүх дугуй 4WD", "Бүх дугуй 4WD", "Урдаа FWD", "Урдаа FWD", "Бүх дугуй 4WD", "Урдаа FWD"],
    "Mileage": [56000, 800000, 100000, 153000, 180, 256000]
}

# Preprocess the sample data
data_tensor = preprocess_sample_data(data)

# Predict prices
model.eval()
with torch.no_grad():
    predicted_prices = model(data_tensor)
    print(predicted_prices)
