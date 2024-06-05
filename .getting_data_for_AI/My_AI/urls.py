from django.urls import path, include
from .views import car_data_view
 
urlpatterns = [
   path('car_data/', car_data_view, name='car_data_view'),
]
