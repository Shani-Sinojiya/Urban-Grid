import cv2
import socket
import json
import threading
import torch
import numpy as np
import os
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from django.core.files.base import ContentFile
from geopy.geocoders import Nominatim
import geocoder
from twilio.rest import Client
import time
from datetime import datetime
from filterpy.kalman import KalmanFilter

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trafficguardian.settings')
django.setup()

from traffic.models import TrafficSignal, AccidentAlert

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path="C:\\Users\\ABC\\Downloads\\yolov5s (1).pt")

# Video sources
video_sources = [
    "C:\\Users\\ABC\\Desktop\\ACCIDENT-DETECTION-WITH-A-REPORTING-SYSTEM-main\\Accident-1.mp4",
]

# Initialize vehicle counts and trackers
total_vehicle_counts = [0]*len(video_sources)
vehicle_tracking_ids = [0]*len(video_sources)
trackers = [[] for _ in range(len(video_sources))]
tracker_dicts = [{} for _ in range(len(video_sources))]
tracker_bboxes = [{} for _ in range(len(video_sources))]
tracker_misses = [{} for _ in range(len(video_sources))]

# Initialize sockets for communication
server_address = ('localhost', 12345)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(('localhost', 12348))

# Variable to store traffic signal status
traffic_signal = "Red"

def generate_unique_id(camera_index):
    global vehicle_tracking_ids
    vehicle_tracking_ids[camera_index] += 1
    return vehicle_tracking_ids[camera_index]

def is_vehicle_already_tracked(camera_index, new_bbox):
    for bbox in tracker_bboxes[camera_index].values():
        if (bbox[0] < new_bbox[0] + new_bbox[2] and
            bbox[0] + bbox[2] > new_bbox[0] and
            bbox[1] < new_bbox[1] + new_bbox[3] and
            bbox[1] + bbox[3] > new_bbox[1]):
            return True
    return False

def filter_boxes(boxes, min_size=1000):
    filtered_boxes = []
    for box in boxes:
        x1, y1, x2, y2, conf = map(int, box[:5])
        width = x2 - x1
        height = y2 - y1
        if width * height > min_size:
            filtered_boxes.append(box)
    return np.array(filtered_boxes)

def send_vehicle_counts(vehicle_counts):
    for i, count in enumerate(vehicle_counts):
        message = json.dumps({'lane_id': f'signal{i+1}', 'vehicle_count': count}).encode('utf-8')
        sock.sendto(message, server_address)

def receive_traffic_signal():
    global traffic_signal
    while True:
        try:
            data, _ = recv_sock.recvfrom(1024)
            message = json.loads(data.decode('utf-8'))
            if message['lane_id'] in [f'signal{i+1}' for i in range(len(video_sources))]:
                traffic_signal = message['traffic_signal']
        except json.JSONDecodeError:
            print("Error decoding JSON message.")
        except Exception as e:
            print(f"Error receiving traffic signal: {e}")

# Start a thread to continuously receive traffic signal status
thread = threading.Thread(target=receive_traffic_signal)
thread.daemon = True
thread.start()

# Create video capture objects for all sources
caps = [cv2.VideoCapture(source) for source in video_sources]

# Use updated tracker (assuming cv2.TrackerCSRT_create is the latest one available)
tracker_factory = cv2.TrackerCSRT_create

# Margin to consider a vehicle out of view
margin = 25
frame_skip = 15  # Process every 5th frame (change as needed)
frame_counter = 0

class AccidentDetection:
    def __init__(self):
        self.model_path = "C:\\Users\\ABC\\Desktop\\accident_detection_model.h5"
        self.dataset_dir = "C:\\Users\\ABC\\Desktop\\your_dataset_directory"  # Update this path
        self.geo_loc = Nominatim(user_agent="GetLoc")
        self.client = Client('TWILIO_SID', 'TWILIO_SECRET')
        self.base_model, self.model = self.load_or_train_model()
        self.last_accident_time = None  # To keep track of the last detected accident time
        self.cooldown_period = 60  # Cooldown period in seconds
        self.confidence_threshold = 0.92  # Increased confidence threshold
        self.confidence_history = []  # To store confidence scores for smoothing
        self.history_length = 50  # Number of frames to consider for smoothing

    def load_or_train_model(self):
        if os.path.exists(self.model_path):
            model = load_model(self.model_path)
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        else:
            base_model, model = self.create_model()
            X_train, X_valid, y_train, y_valid = self.prepare_data()
            X_train_features = base_model.predict(X_train).reshape(X_train.shape[0], -1)
            X_valid_features = base_model.predict(X_valid).reshape(X_valid.shape[0], -1)
            model.fit(X_train_features, y_train, epochs=100, validation_data=(X_valid_features, y_valid))
            self.save_model(model)
        return base_model, model

    def create_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model = Sequential()
        model.add(InputLayer((7*7*512,)))
        model.add(Dense(units=1024, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return base_model, model

    def save_model(self, model):
        model.save(self.model_path)

    def prepare_data(self):
        train_dir = "C:\\Users\\ABC\\Desktop\\Accident-Detection-System-main\\data\\train"
        val_dir = "C:\\Users\\ABC\\Desktop\\Accident-Detection-System-main\\data\\val"
        test_dir = "C:\\Users\\ABC\\Desktop\\Accident-Detection-System-main\\data\\test"

        # Use ImageDataGenerator to load and preprocess images
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
        val_generator = datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
        test_generator = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

        X_train, y_train = self.load_data_from_generator(train_generator)
        X_val, y_val = self.load_data_from_generator(val_generator)
        X_test, y_test = self.load_data_from_generator(test_generator)

        return X_train, X_val, y_train, y_val
    
    def load_data_from_generator(self, generator):
        X = []
        y = []
        for _ in range(len(generator)):
            X_batch, y_batch = generator.next()
            X.append(X_batch)
            y.append(y_batch)
        return np.vstack(X), np.vstack(y)   

    def detect_accident(self, frame):
        try:
            resized_frame = cv2.resize(frame, (224, 224))  # Resize frame to match VGG16 input
            frame_array = np.expand_dims(resized_frame, axis=0) / 255.0  # Normalize
            features = self.base_model.predict(frame_array)
            features_flattened = features.reshape(1, -1)
            prediction = self.model.predict(features_flattened)
            confidence = prediction[0][1]  # Assuming the second class is 'accident'
            return confidence
        except Exception as e:
            print(f"Error detecting accident: {e}")
            return 0

    def process_frame(self, frame):
        current_time = time.time()
        try:
            accident_confidence = self.detect_accident(frame)
            self.confidence_history.append(accident_confidence)
            if len(self.confidence_history) > self.history_length:
                self.confidence_history.pop(0)
            smoothed_confidence = np.mean(self.confidence_history)
            print(f"Smoothed Accident Confidence: {smoothed_confidence}")

            if smoothed_confidence > self.confidence_threshold:
                if self.last_accident_time is None or (current_time - self.last_accident_time) > self.cooldown_period:
                    print("Accident Detected!")
                    try:
                        locname = self.geo_loc.reverse(geocoder.ip('me').latlng)
                        message = self.client.messages.create(
                            body=f"Accident detected at location: {locname}!",
                            from_='+19388391544', 
                            to='+916353940369'
                        )
                        print(f"Message sent with SID: {message.sid}")
                    except Exception as e:
                        print(f"Failed to send message: {e}")
                    self.last_accident_time = current_time
        except Exception as e:
            print(f"Error processing frame: {e}")

accident_detection = AccidentDetection()

while True:
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_counter += 1
        if frame_counter % frame_skip == 0:
            # Perform YOLOv5 detection
            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()
            filtered_boxes = filter_boxes(detections, min_size=1000)

            # Update trackers and count vehicles
            new_trackers = []
            for box in filtered_boxes:
                x1, y1, x2, y2, conf = map(int, box[:5])
                bbox = (x1, y1, x2-x1, y2-y1)
                if not is_vehicle_already_tracked(i, bbox):
                    new_tracker = tracker_factory()
                    new_tracker.init(frame, bbox)
                    tracker_id = generate_unique_id(i)
                    tracker_dicts[i][tracker_id] = new_tracker
                    tracker_bboxes[i][tracker_id] = bbox
                    tracker_misses[i][tracker_id] = 0
                    new_trackers.append(new_tracker)

            # Track existing vehicles
            to_remove = []
            for tracker_id, tracker in tracker_dicts[i].items():
                success, bbox = tracker.update(frame)
                if success:
                    bbox = tuple(map(int, bbox))
                    if not is_vehicle_already_tracked(i, bbox):
                        tracker_bboxes[i][tracker_id] = bbox
                        tracker_misses[i][tracker_id] = 0
                else:
                    tracker_misses[i][tracker_id] += 1
                    if tracker_misses[i][tracker_id] > 5:  # Adjust threshold as needed
                        to_remove.append(tracker_id)

            # Remove lost trackers
            for tracker_id in to_remove:
                del tracker_dicts[i][tracker_id]
                del tracker_bboxes[i][tracker_id]

            # Count detected vehicles
            vehicle_count = len(tracker_dicts[i])
            total_vehicle_counts[i] = vehicle_count
            send_vehicle_counts(total_vehicle_counts)

            # Process accident detection
            accident_detection.process_frame(frame)

        # Display the frame (for debugging purposes)
        cv2.imshow(f'Video {i}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
