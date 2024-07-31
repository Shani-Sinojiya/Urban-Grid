import sys
import os
import django
import socket
import json
import time
import threading
from datetime import datetime
from django.utils import timezone
from django.core.files.base import ContentFile

# Set up Django settings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'trafficguardian')))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trafficguardian.settings')
django.setup()

from traffic.models import TrafficSignal, AccidentAlert

class CentralController:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('localhost', 12345))
        self.signals_data = {}
        self.lock = threading.Lock()
        self.signal_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.signal_address = ('localhost', 5001)
        self.green_signal_end_time = {}  # Track end times for green signals

    def receive_data(self):
        while True:
            data, _ = self.sock.recvfrom(1024)
            message = json.loads(data.decode('utf-8'))
            lane_id = message['lane_id']
            vehicle_count = message['vehicle_count']
            TrafficSignal.objects.update_or_create(
                lane_id=lane_id,
                defaults={'vehicle_count': vehicle_count}
            )
            with self.lock:
                self.signals_data[lane_id] = vehicle_count
            print(f"Received data from {lane_id}: {vehicle_count}")

    def send_signal_state(self, lane_id, traffic_signal, vehicle_count):
        message = json.dumps({
            'lane_id': lane_id,
            'traffic_signal': traffic_signal,
            'vehicle_count': vehicle_count
        }).encode('utf-8')
        self.signal_socket.sendto(message, self.signal_address)
        print(f'Sent updated signal state: {{\'lane_id\': \'{lane_id}\', \'traffic_signal\': \'{traffic_signal}\', \'vehicle_count\': {vehicle_count}}}')

    def update_traffic_signals(self):
        try:
            if not self.signals_data:
                return  # No data to process

            with self.lock:
                # Find the maximum and second-highest vehicle count
                sorted_counts = sorted(self.signals_data.values(), reverse=True)
                max_count = sorted_counts[0] if sorted_counts else None
                second_max_count = sorted_counts[1] if len(sorted_counts) > 1 else None

                # Find all lanes with the maximum and second-highest counts
                max_lane_ids = [lane_id for lane_id, count in self.signals_data.items() if count == max_count]
                second_max_lane_ids = [lane_id for lane_id, count in self.signals_data.items() if count == second_max_count]

            traffic_signals = {}
            for lane_id in self.signals_data:
                if lane_id in max_lane_ids:
                    traffic_signals[lane_id] = 'Green'
                elif lane_id in second_max_lane_ids:
                    traffic_signals[lane_id] = 'Yellow'
                else:
                    traffic_signals[lane_id] = 'Red'

            # Update TrafficSignal model in the database
            for lane_id, signal in traffic_signals.items():
                if signal == 'Green':
                    # Set end time for green signal
                    self.green_signal_end_time[lane_id] = timezone.now() + timezone.timedelta(seconds=30)  # Adjust time as needed
                else:
                    # Reset end time for non-green signals
                    self.green_signal_end_time.pop(lane_id, None)

                TrafficSignal.objects.update_or_create(
                    lane_id=lane_id,
                    defaults={
                        'traffic_signal': signal,
                        'vehicle_count': self.signals_data[lane_id]
                    }
                )
                self.send_signal_state(lane_id, signal, self.signals_data[lane_id])

            # Reset vehicle count for lanes where green signal timer has expired
            self.reset_vehicle_count()

        except Exception as e:
            print(f"Error updating traffic signals: {e}")


    def reset_vehicle_count(self):
        current_time = timezone.now()
        for lane_id, end_time in list(self.green_signal_end_time.items()):
            if current_time >= end_time:
                # Reset vehicle count to 0 and send update
                TrafficSignal.objects.update_or_create(
                    lane_id=lane_id,
                    defaults={'vehicle_count': 0}
                )
                self.signals_data[lane_id] = 0
                self.send_signal_state(lane_id, 'Green', 0)
                # Remove lane_id from green_signal_end_time
                self.green_signal_end_time.pop(lane_id, None)

    def save_accident_alert(self, location, image_path):
        alert = AccidentAlert(location=location)
        with open(image_path, 'rb') as img_file:
            alert.image.save(os.path.basename(image_path), ContentFile(img_file.read()))
        alert.save()
        print(f"Saved accident alert: {location}")

    def run(self):
        # Start thread to receive data
        recv_thread = threading.Thread(target=self.receive_data)
        recv_thread.daemon = True
        recv_thread.start()

        while True:
            self.update_traffic_signals()
            time.sleep(5)

if __name__ == "__main__":
    controller = CentralController()
    controller.run()
