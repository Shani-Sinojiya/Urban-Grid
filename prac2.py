import os
import django
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trafficguardian.settings')

# Setup Django
django.setup()

import cv2
import socket
import json
import threading
import torch
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from geopy.geocoders import Nominatim
import geocoder
from twilio.rest import Client
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from django.conf import settings
from traffic.models import Accident  # Import your model here

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path="C:\\Users\\ABC\\Downloads\\yolov5s (1).pt")

# Video sources
video_sources = [
    "C:\\Users\\ABC\\Downloads\\video.mp4",
    "C:\\Users\\ABC\\Desktop\\ACCIDENT-DETECTION-WITH-A-REPORTING-SYSTEM-main\\Accident-1.mp4",
]

# Initialize vehicle counts and trackers
total_vehicle_counts = [0, 0]
vehicle_tracking_ids = [0, 0]
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
            logging.error("Error decoding JSON message.")
        except Exception as e:
            logging.error(f"Error receiving traffic signal: {e}")

# Start a thread to continuously receive traffic signal status
thread = threading.Thread(target=receive_traffic_signal)
thread.daemon = True
thread.start()

# Create video capture objects for all sources
caps = [cv2.VideoCapture(source) for source in video_sources]

# Use CSRT tracker for better accuracy
tracker_factory = cv2.TrackerCSRT_create

# Margin to consider a vehicle out of view
margin = 15

class AccidentDetection:
    def __init__(self):
        self.model_path = "C:\\Users\\ABC\\Desktop\\accident_detection_model.h5"
        self.dataset_dir = "C:\\Users\\ABC\\Desktop\\your_dataset_directory"  # Update this path
        self.geo_loc = Nominatim(user_agent="GetLoc")
        self.client = Client('TWILIO_SID', 'TWILIO_SECRET')
        self.base_model, self.model = self.load_or_train_model()
        self.last_accident_time = None  # To keep track of the last detected accident time
        self.cooldown_period = 60  # Cooldown period in seconds
        self.confidence_threshold = 0.91  # Increased confidence threshold
        self.confidence_history = []  # To store confidence scores for smoothing
        self.history_length = 5  # Number of frames to consider for smoothing

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
            X.extend(X_batch)
            y.extend(y_batch)
        return np.array(X), np.array(y)

    def save_accident_frame(self, frame, frame_count):
        filename = f"accident_{frame_count}.jpg"
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, 'accidents', filename), frame)
        return filename

    def process_frame(self, frame):
        img = resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        current_time = time.time()
        confidence = self.model.predict(self.base_model.predict(img).reshape(1, 7*7*512))[0][1]

        accident_confidence = self.model.predict_accident(frame)
        self.confidence_history.append(accident_confidence)
        if len(self.confidence_history) > self.history_length:
            self.confidence_history.pop(0)

        smoothed_confidence = np.mean(self.confidence_history)

        if smoothed_confidence > self.confidence_threshold:
            if self.last_accident_time is None or (current_time - self.last_accident_time > self.cooldown_period):
                try:
                    frame_count = len(os.listdir(os.path.join(settings.MEDIA_ROOT, 'accidents')))
                    filename = self.save_accident_frame(frame, frame_count)

                    # Update the Django database
                    Accident.objects.create(
                        frame=filename,
                        detected_at=current_time,
                    )

                    self.last_accident_time = current_time

                    # Send notification
                    message = self.client.messages.create(
                        body=f"Accident detected at {datetime.now()}.",
                        from_='+12132910084',
                        to='+91 73034 79075'
                    )
                    print(message.sid)

                    # Return True if an accident is detected
                    return True
                except Exception as e:
                    logging.error(f"Error in processing frame: {e}")
                    return False
        return False

def process_video(camera_index, video_source, accident_detection):
    global total_vehicle_counts, trackers, tracker_dicts, tracker_bboxes, tracker_misses
    try:
        cap = cv2.VideoCapture(video_source)
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'output_camera_{camera_index}.mp4', fourcc, 30, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to read frame from video source {video_source}")
                break

            # Perform vehicle detection
            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()

            # Update trackers
            for id, tracker in list(trackers[camera_index].items()):
                success, bbox = tracker.update(frame)
                if success:
                    tracker_bboxes[camera_index][id] = bbox
                    tracker_misses[camera_index][id] = 0
                else:
                    tracker_misses[camera_index][id] += 1
                    if tracker_misses[camera_index][id] > margin:
                        del trackers[camera_index][id]
                        del tracker_bboxes[camera_index][id]
                        del tracker_misses[camera_index][id]
                        total_vehicle_counts[camera_index] -= 1

            # Add new detections to trackers
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == 2 and conf > 0.3:
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    if not is_vehicle_already_tracked(camera_index, bbox):
                        tracker = tracker_factory()
                        tracker.init(frame, tuple(bbox))
                        new_id = generate_unique_id(camera_index)
                        trackers[camera_index][new_id] = tracker
                        tracker_bboxes[camera_index][new_id] = bbox
                        tracker_misses[camera_index][new_id] = 0
                        total_vehicle_counts[camera_index] += 1

            # Draw bounding boxes and IDs
            for id, bbox in tracker_bboxes[camera_index].items():
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, str(id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            # Process frame for accident detection
            if accident_detection.process_frame(frame):
                logging.info(f"Accident detected in camera {camera_index}")

            out.write(frame)
            cv2.imshow(f'Camera {camera_index}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    except cv2.error as e:
        logging.error(f"OpenCV error: {e}")
    except Exception as e:
        logging.error(f"Error processing video from {video_source}: {e}")

if __name__ == "__main__":
    accident_detection = AccidentDetection()

    # Start video processing threads
    threads = []
    for i, source in enumerate(video_sources):
        thread = threading.Thread(target=process_video, args=(i, source, accident_detection))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Send vehicle counts periodically
    while True:
        send_vehicle_counts(total_vehicle_counts)
        time.sleep(5)
