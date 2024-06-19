from django.shortcuts import render
from .forms import SelectionForm
from django.http import JsonResponse
from .models import Mark, MotorBagtaamj, Xrop, Joloo, UildverlesenOn, OrjIrsenOn, Hutlugch, YavsanKm
from .training.model_training import preprocess_sample_data
import joblib
import os
import logging

logger = logging.getLogger(__name__)

# Load pre-trained models and scalers (adjust paths accordingly)
xgboost_model_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\xgboost_model.joblib'
labelencoder_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\labelencoder.joblib'
scaler_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\scaler.joblib'
target_scaler_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\target_scaler.joblib'
poly_path = r'C:\Users\dvkka\Documents\AI-test-training\getting_data_for_AI\.getting_data_for_AI\poly.joblib'

loaded_xgboost_model = joblib.load(xgboost_model_path)
label_encoders = joblib.load(labelencoder_path)
scaler = joblib.load(scaler_path)
target_scaler = joblib.load(target_scaler_path)
poly = joblib.load(poly_path)

# Function to preprocess user input and make predictions
def predict_car_price(user_input):
    try:
        # Ensure all values in user_input are lists
        for key in user_input:
            if not isinstance(user_input[key], list):
                user_input[key] = [user_input[key]]

        # Log the input data
        logger.debug(f"User input after conversion to list: {user_input}")

        # Preprocess user input
        features_tensor = preprocess_sample_data(user_input, label_encoders, scaler, poly)

        # Make predictions
        sample_pred = loaded_xgboost_model.predict(features_tensor.numpy())

        # Inverse transform the predicted price
        predicted_price = target_scaler.inverse_transform(sample_pred.reshape(-1, 1))[0][0]

        return predicted_price
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

# Django view to handle form submission and prediction
def predict_price_view(request):
    if request.method == 'POST':
        form = SelectionForm(request.POST)
        if form.is_valid():
            # Extract form data and ensure all values are lists
            sample_data = {
                'Uildverlegch': [form.cleaned_data['uildverlegch']],
                'Mark': [form.cleaned_data['mark']],
                'Xrop': [form.cleaned_data['xrop']],
                'Joloo': [form.cleaned_data['joloo']],
                'Hudulguur': [form.cleaned_data['hudulguur']],
                'Hutlugch': [form.cleaned_data['hutlugch']],
                'Motor_bagtaamj': [form.cleaned_data['motor_bagtaamj']],
                'Uildverlesen_on': [form.cleaned_data['uildverlesen_on']],
                'Orj_irsen_on': [form.cleaned_data['orj_irsen_on']],
                'Yavsan_km': [form.cleaned_data['yavsan_km']]
            }

            # Predict car price
            predicted_price = predict_car_price(sample_data)

            if predicted_price is not None:
                # Render prediction.html with predicted price
                return render(request, 'prediction.html', {'predicted_price': predicted_price})
            else:
                # Handle prediction error directly (print or log the error)
                return render(request, 'selection.html', {'form': form, 'error_message': 'Failed to predict price. Please try again.'})
    else:
        form = SelectionForm()

    return render(request, 'selection.html', {'form': form})

# AJAX views for loading dropdown options
def load_mark(request):
    uildverlegch_id = request.GET.get('uildverlegch')
    marks = Mark.objects.filter(uildverlegch_id=uildverlegch_id).order_by('name')
    return JsonResponse(list(marks.values('id', 'name')), safe=False)

def load_motor_bagtaamj(request):
    mark_id = request.GET.get('mark')
    motor_bagtaamj = MotorBagtaamj.objects.filter(mark_id=mark_id).order_by('size')
    return JsonResponse(list(motor_bagtaamj.values('id', 'size')), safe=False)

def load_xrop(request):
    motor_bagtaamj_id = request.GET.get('motor_bagtaamj')
    xrop = Xrop.objects.filter(motor_bagtaamj_id=motor_bagtaamj_id).order_by('type')
    return JsonResponse(list(xrop.values('id', 'type')), safe=False)

def load_joloo(request):
    xrop_id = request.GET.get('xrop')
    joloo = Joloo.objects.filter(xrop_id=xrop_id).order_by('type')
    return JsonResponse(list(joloo.values('id', 'type')), safe=False)

def load_uildverlesen_on(request):
    joloo_id = request.GET.get('joloo')
    uildverlesen_on = UildverlesenOn.objects.filter(joloo_id=joloo_id).order_by('year')
    return JsonResponse(list(uildverlesen_on.values('id', 'year')), safe=False)

def load_orj_irsen_on(request):
    uildverlesen_on_id = request.GET.get('uildverlesen_on')
    orj_irsen_on = OrjIrsenOn.objects.filter(uildverlesen_on_id=uildverlesen_on_id).order_by('year')
    return JsonResponse(list(orj_irsen_on.values('id', 'year')), safe=False)

def load_hutlugch(request):
    orj_irsen_on_id = request.GET.get('orj_irsen_on')
    hutlugch = Hutlugch.objects.filter(orj_irsen_on_id=orj_irsen_on_id).order_by('type')
    return JsonResponse(list(hutlugch.values('id', 'type')), safe=False)

def load_yavsan_km(request):
    hutlugch_id = request.GET.get('hutlugch')
    yavsan_km = YavsanKm.objects.filter(hutlugch_id=hutlugch_id).order_by('distance')
    return JsonResponse(list(yavsan_km.values('id', 'distance')), safe=False)
