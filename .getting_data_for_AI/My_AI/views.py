from django.shortcuts import render
from data_retrive.models import CarData 
from django.http import HttpResponse
import pandas as pd
import re

def clean_engine_capacity(engine_capacity):
    if isinstance(engine_capacity, str):
        match = re.search(r'\d+(\.\d+)?', engine_capacity)
        return float(match.group()) if match else None
    return engine_capacity

def clean_mileage(mileage):
    if isinstance(mileage, str):
        match = re.search(r'\d+', mileage.replace(',', ''))
        return int(match.group()) if match else None
    return mileage

def car_data_view(request):
    car_data_objects = CarData.objects.all()

    data = {
        'Manufacturer': [],
        'Model': [],
        'Posted Date': [],
        'Engine Capacity': [],
        'Transmission': [],
        'Steering': [],
        'Type': [],
        'Color': [],
        'Manufactured Year': [],
        'Import Year': [],
        'Drive': [],
        'Interior Color': [],
        'Drive Type': [],
        'Mileage': [],
        'Price': [],
        'Unique ID': []
    }

    for car_data in car_data_objects:
        data['Manufacturer'].append(car_data.manufacturer)
        data['Model'].append(car_data.model)
        data['Posted Date'].append(car_data.posted_date)
        data['Engine Capacity'].append(clean_engine_capacity(car_data.engine_capacity))
        data['Transmission'].append(car_data.transmission)
        data['Steering'].append(car_data.steering)
        data['Type'].append(car_data.type)
        data['Color'].append(car_data.color)
        data['Manufactured Year'].append(car_data.manufacture_year)
        data['Import Year'].append(car_data.import_year)
        data['Drive'].append(car_data.drive)
        data['Interior Color'].append(car_data.interior_color)
        data['Drive Type'].append(car_data.drive_type)
        data['Mileage'].append(clean_mileage(car_data.mileage))
        data['Price'].append(car_data.price)
        data['Unique ID'].append(car_data.unique_id)


    df = pd.DataFrame(data)


    df['Engine Capacity'].fillna(df['Engine Capacity'].mean(), inplace=True)


    df = df[(df['Mileage'] < 1000000) & (df['Mileage'] > 0)]  # Example filter


    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename="car_data.xlsx"'

    # Write the DataFrame to the Excel file
    df.to_excel(response, index=False, engine='xlsxwriter')

    return response
