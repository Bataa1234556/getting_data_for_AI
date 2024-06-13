from django.urls import path, include
from . import views


app_name = "My_AI"

urlpatterns = [
    path('', views.selection_view, name='selection_view'),
    path('ajax/load-mark/', views.load_mark, name='ajax_load_mark'),
    path('ajax/load-motor-bagtaamj/', views.load_motor_bagtaamj, name='ajax_load_motor_bagtaamj'),
    path('ajax/load-xrop/', views.load_xrop, name='ajax_load_xrop'),
    path('ajax/load-joloo/', views.load_joloo, name='ajax_load_joloo'),
    path('ajax/load-uildverlesen-on/', views.load_uildverlesen_on, name='ajax_load_uildverlesen_on'),
    path('ajax/load-orj-irsen-on/', views.load_orj_irsen_on, name='ajax_load_orj_irsen_on'),
    path('ajax/load-hutlugch/', views.load_hutlugch, name='ajax_load_hutlugch'),
    path('ajax/load-yavsan-km/', views.load_yavsan_km, name='ajax_load_yavsan_km'),
    path('predict/', views.predict_price_view, name='predict_price'),
]