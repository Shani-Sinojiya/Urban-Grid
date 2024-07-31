# traffic/views.py
from django.shortcuts import render
from .models import TrafficSignal, AccidentAlert
from django.http import JsonResponse
import requests

def lane_list(request):
    signals = TrafficSignal.objects.all()
    accident_alert = request.session.pop('accident_alert', None)
    accident_image = None

    # Fetch the latest accident alert
    latest_accident = AccidentAlert.objects.order_by('-time').first()
    if latest_accident:
        accident_alert = "Accident detected!"
        # Check if the image field is not None before accessing the URL
        if latest_accident.frame and hasattr(latest_accident.frame, 'url'):
            accident_image = latest_accident.frame.url
        else:
            accident_image = None  # or a default image URL if you prefer

    # Calculate timer durations
    for signal in signals:
        count = signal.vehicle_count
        if 1 <= count <= 5:
            signal.timer_duration = 30
        elif 6 <= count <= 15:
            signal.timer_duration = 90
        elif 16 <= count <= 30:
            signal.timer_duration = 120
        else:
            signal.timer_duration = 0  # Or any default value

    return render(request, 'traffic/lane_list.html', {
        'signals': signals,
        'accident_alert': accident_alert,
        'accident_frame': accident_image,
    })

def get_signals(request):
    signals = TrafficSignal.objects.all()
    signals_data = []
    for signal in signals:
        signals_data.append({
            'lane_id': signal.lane_id,
            'lane_number': signal.lane_number,
            'current_signal': signal.traffic_signal,
            'vehicle_count': signal.vehicle_count,
            'timer_duration': signal.timer_duration
        })
    print("Signals data:", signals_data)  # Debugging statement
    return JsonResponse({'signals': signals_data})

def fetch_from_central_controller(lane_id):
    try:
        response = requests.get(f'http://localhost:5001/api/signals/{lane_id}')
        data = response.json()
        return data.get('current_signal', 'Unknown')
    except requests.RequestException as e:
        print(f"Error fetching signal: {e}")
        return 'Unknown'
