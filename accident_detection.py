import os
import cv2
import math
import geocoder
import pandas as pd
from twilio.rest import Client
from geopy.geocoders import Nominatim
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from skimage.transform import resize
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.model_selection import train_test_split
import django
import sys
# Setup Django environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'trafficguardian')))

# Set up Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trafficguardian.settings')

django.setup()

from traffic.models import Accident
from django.contrib.sessions.models import Session

def capture_frames(video_file, output_prefix, output_dir):
    count = 0
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_file}")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % math.floor(frame_rate) == 0:
            filename = os.path.join(output_dir, f"{output_prefix}{count}.jpg")
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()
    print("Frame capture completed.")

def load_and_preprocess_images(image_ids, image_folder=''):
    images = []
    for img_name in image_ids:
        img_path = os.path.join(image_folder, f"frame{img_name}")
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found.")
            continue
        img = plt.imread(img_path)
        img_resized = resize(img, preserve_range=True, output_shape=(224, 224)).astype(int)
        images.append(img_resized)
    return np.array(images)

def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential()
    model.add(InputLayer((7*7*512,)))
    model.add(Dense(units=1024, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return base_model, model

def save_model(model, model_path):
    model.save(model_path)

def load_model_from_file(model_path):
    return load_model(model_path)

def main():
    frame_dir = "C:\\Users\\ABC\\Desktop\\ACCIDENT-DETECTION-WITH-A-REPORTING-SYSTEM-main\\frames"
    test_frame_dir = "C:\\Users\\ABC\\Desktop\\ACCIDENT-DETECTION-WITH-A-REPORTING-SYSTEM-main\\test_frames"
    model_path = "C:\\Users\\ABC\\Desktop\\accident_detection_model.h5"

    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(test_frame_dir, exist_ok=True)

    try:
        capture_frames("C:\\Users\\ABC\\Desktop\\ACCIDENT-DETECTION-WITH-A-REPORTING-SYSTEM-main\\Accidents.mp4", "frame", frame_dir)
    except Exception as e:
        print(f"Error capturing frames: {e}")
        return

    try:
        data = pd.read_csv("C:\\Users\\ABC\\Desktop\\ACCIDENT-DETECTION-WITH-A-REPORTING-SYSTEM-main\\mapping.csv")
        X = load_and_preprocess_images(data.Image_ID, frame_dir)
        if X.size == 0:
            raise Exception("No images loaded for training.")
        y = to_categorical(data.Class)
        X = preprocess_input(X)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
        
        if os.path.exists(model_path):
            model = load_model_from_file(model_path)
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        else:
            base_model, model = create_model()
            X_train_features = base_model.predict(X_train).reshape(X_train.shape[0], -1)
            X_valid_features = base_model.predict(X_valid).reshape(X_valid.shape[0], -1)
            
            model.fit(X_train_features, y_train, epochs=100, validation_data=(X_valid_features, y_valid))
            save_model(model, model_path)
    except Exception as e:
        print(f"Error processing training data: {e}")
        return

    try:
        capture_frames("C:\\Users\\ABC\\Desktop\\acc.mp4", "frametest", test_frame_dir)
        test_data = pd.read_csv("C:\\Users\\ABC\\Desktop\\ACCIDENT-DETECTION-WITH-A-REPORTING-SYSTEM-main\\test.csv")
        if test_data.empty:
            raise Exception("Test data CSV is empty.")
        test_images = load_and_preprocess_images(test_data.Image_ID, test_frame_dir)
        if test_images.size == 0:
            raise Exception("No test images loaded.")
        test_images = preprocess_input(test_images)
        test_features = base_model.predict(test_images).reshape(test_images.shape[0], -1)
        test_features /= test_features.max()
        
        predictions = model.predict(test_features)
        for i in range(len(predictions)):
            print("No Accident" if predictions[i][0] < predictions[i][1] else "Accident")
    except Exception as e:
        print(f"Error processing test data: {e}")
        return

    try:
        geo_loc = Nominatim(user_agent="GetLoc")
        g = geocoder.ip('me')
        locname = geo_loc.reverse(g.latlng)

        account_sid = 'TWILIO_SID'
        auth_token = 'TWILIO_SECRET'
        client = Client(account_sid, auth_token)

        cap = cv2.VideoCapture("C:\\Users\\ABC\\Desktop\\acc.mp4")
        i = 0
        flag = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if predictions[int(i/15) % len(predictions)][0] < predictions[int(i/15) % len(predictions)][1]:
                predict = "No Accident"
            else:
                predict = "Accident"
                flag = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, predict, (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_4)
            cv2.imshow('Frame', frame)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if flag == 1:
            client.messages.create(
                body=f"Accident detected in {locname.address}",
                from_='+17622494914',
                to='+916206709549'
            )
            
            accident = Accident(location=locname.address, details="Accident detected.")
            accident.save()

            session = Session.objects.create()
            session['accident_alert'] = f"Accident detected in {locname.address}"
            session.save()
    except Exception as e:
        print(f"Error in notification process: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
