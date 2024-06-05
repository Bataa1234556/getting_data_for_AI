import torch
import pandas as pd
from model_training import CarPriceModel  # Import the trained model from trained_model.py

# Define the synthetic data
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


def preprocessing_sample_data(data):
    data_tensor = torch.tensor([
        data["Engine Capacity"],
        data["Manufactured Year"],
        data["Import Year"],
        data["Mileage"],
    ])


    data_tensor = data_tensor.transpose(0,1)
    return data_tensor

data_tensor = preprocessing_sample_data(data)

