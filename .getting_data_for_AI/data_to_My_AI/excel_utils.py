import pandas as pd
from .models import SourceModel

def read_excel(file_path):
    df = pd.read_excel(file_path)
    for _, row in df.iterrows():
        SourceModel.objects.create(
            uildverlegch=row['Uildverlegch'],
            mark=row['Mark'],
            motor_bagtaamj=row['Motor_bagtaamj'],
            xrop=row['Xrop'],
            joloo=row['Joloo'],
            uildverlesen_on=row['Uildverlesen_on'],
            orj_irsen_on=row['Orj_irsen_on'],
            hudulguur=row['Hudulguur'],
            hutlugch=row['Hutlugch'],
            yavsan_km=row['Yavsan_km'],
            une=row['Une']
        )
