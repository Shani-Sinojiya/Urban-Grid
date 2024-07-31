from django.db import models

class TrafficSignal(models.Model):
    lane_id = models.CharField(max_length=20, unique=True)
    traffic_signal = models.CharField(max_length=10)
    vehicle_count = models.IntegerField()

    def __str__(self):
        return f'{self.lane_id} - {self.traffic_signal}'

class AccidentAlert(models.Model):
    location = models.CharField(max_length=255)
    frame = models.ImageField(upload_to='accidents_frames/', blank=True, null=True)
    time = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f'{self.location} - {self.timestamp}'
