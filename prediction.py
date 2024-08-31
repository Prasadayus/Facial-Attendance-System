from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from threading import Thread

app = Flask(__name__)

# Load the model and label encoder
def load_resources(model_path, label_encoder_path):
    model = load_model(model_path)
    with open(label_encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)
    return model, label_encoder

model_path = r'densenet121_face_recognition_model.h5'
label_encoder_path = r'label_encoder_densenet.pkl'
model, label_encoder = load_resources(model_path, label_encoder_path)

# Preprocess image function
def preprocess_image(img, target_size=(224, 224)):
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = img
    else:
        raise ValueError("Unexpected number of channels in the input image")

    img_resized = cv2.resize(img_rgb, target_size)
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    return img_expanded

# Function to predict and log results
def predict_and_log(img, model, label_encoder, log_filename='logs/prediction_log1.xlsx', entry_interval_minutes=15):
    img_preprocessed = preprocess_image(img)
    
    try:
        prediction = model.predict(img_preprocessed)
        predicted_label_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        if not os.path.exists(log_filename):
            df = pd.DataFrame(columns=['Name', 'Timestamp'])
            last_entry = {}
        else:
            df = pd.read_excel(log_filename)
            if 'Name' not in df.columns:
                return "Unknown"
            last_entry = df.set_index('Name')['Timestamp'].to_dict()
        
        if predicted_label in last_entry:
            last_time = datetime.strptime(last_entry[predicted_label], "%Y-%m-%d %H:%M:%S")
            if datetime.now() - last_time < timedelta(minutes=entry_interval_minutes):
                return f"{predicted_label} (Already Logged)"
        
        new_row = pd.DataFrame({'Name': [predicted_label], 'Timestamp': [timestamp]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(log_filename, index=False)
        
        return predicted_label
    except Exception as e:
        print(f"Error during prediction or logging: {e}")
        return "Unknown"

# Function to capture video and process frames
def capture_and_predict():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while app.config.get('CAPTURE'):
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture image")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_cropped = frame[y:y+h, x:x+w]
            predicted_name = predict_and_log(face_cropped, model, label_encoder)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            text = predicted_name
            color = (0, 255, 0) if "Already Logged" not in predicted_name else (0, 0, 255)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_capture():
    app.config['CAPTURE'] = True
    thread = Thread(target=capture_and_predict)
    thread.start()
    return jsonify({'status': 'Started'})

@app.route('/stop', methods=['POST'])
def stop_capture():
    app.config['CAPTURE'] = False
    return jsonify({'status': 'Stopped'})

if __name__ == '__main__':
    app.config['CAPTURE'] = False
    app.run(debug=True)
