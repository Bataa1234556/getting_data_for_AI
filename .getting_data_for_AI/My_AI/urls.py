from django.urls import path
from .views import (
    predict_price_view,
    load_mark,
    load_motor_bagtaamj,
    load_xrop,
    load_joloo,
    load_uildverlesen_on,
    load_orj_irsen_on,
    load_hutlugch,
    load_yavsan_km
)

urlpatterns = [
    path('', predict_price_view, name='predict_price'),
    path('load-mark/', load_mark, name='load_mark'),
    path('load-motor-bagtaamj/', load_motor_bagtaamj, name='load_motor_bagtaamj'),
    path('load-xrop/', load_xrop, name='load_xrop'),
    path('load-joloo/', load_joloo, name='load_joloo'),
    path('load-uildverlesen-on/', load_uildverlesen_on, name='load_uildverlesen_on'),
    path('load-orj-irsen-on/', load_orj_irsen_on, name='load_orj_irsen_on'),
    path('load-hutlugch/', load_hutlugch, name='load_hutlugch'),
    path('load-yavsan-km/', load_yavsan_km, name='load_yavsan_km')
]
