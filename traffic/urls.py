from django.urls import path
from . import views

urlpatterns = [
    path('api/signals/', views.get_signals, name='get_signals'),
    path('lanes/', views.lane_list, name='lane_list'),
    path('', views.lane_list),  # Redirect root URL to the lane list
]

