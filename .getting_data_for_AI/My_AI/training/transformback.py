import pandas as pd
from sklearn.preprocessing import LabelEncoder  
# Load the provided original mashin_data.xlsx file
original_file_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\My_AI\excel\mashin_data.xlsx'
original_df = pd.read_excel(original_file_path)

# Display the first few rows of the original dataframe to understand its structure
le = LabelEncoder()

# Fit and transform the columns
original_df['Uildverlegch_encoded'] = le.fit_transform(original_df['Uildverlegch'])
uildverlegch_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print('Uildverlegch Mapping:', uildverlegch_mapping)

original_df['Hudulguur_encoded'] = le.fit_transform(original_df['Hudulguur'])
hudulguur_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print('Hudulguur Mapping:', hudulguur_mapping)

original_df['Mark_encoded'] = le.fit_transform(original_df['Mark'])
mark_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print('Mark Mapping:', mark_mapping)

original_df['Hutlugch_encoded'] = le.fit_transform(original_df['Hutlugch'])
hutlugch_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print('Hutlugch Mapping:', hutlugch_mapping)



# Load the transformed version1.xlsx file
transformed_file_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\My_AI\excel\version1.xlsx'
transformed_df = pd.read_excel(transformed_file_path)

# Reverse label encodings
transformed_df['Uildverlegch'] = transformed_df['Uildverlegch'].map({v: k for k, v in uildverlegch_mapping.items()})
transformed_df['Hudulguur'] = transformed_df['Hudulguur'].map({v: k for k, v in hudulguur_mapping.items()})
transformed_df['Mark'] = transformed_df['Mark'].map({v: k for k, v in mark_mapping.items()})
transformed_df['Hutlugch'] = transformed_df['Hutlugch'].map({v: k for k, v in hutlugch_mapping.items()})

# Convert binary columns back to original categorical values
transformed_df['Xrop'] = transformed_df['Xrop'].replace({1: 'Автомат', 0: 'Механик'})
transformed_df['Joloo'] = transformed_df['Joloo'].replace({1: 'Зөв', 0: 'Буруу'})

# Reformat Motor_bagtaamj
transformed_df['Motor_bagtaamj'] = (transformed_df['Motor_bagtaamj'] / 1000).astype(str) + ' л'

# Reformat Yavsan_km
transformed_df['Yavsan_km'] = transformed_df['Yavsan_km'].astype(str) + ' км'

# Save the restored dataframe to a new Excel file
restored_file_path = 'restored_version.xlsx'
transformed_df.to_excel(restored_file_path, index=False)
