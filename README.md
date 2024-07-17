# Facial Attendance System

Facial Attendance System is a project aimed at automating attendance management using facial recognition. This system detects faces in real-time using a webcam, recognizes the faces of individuals enrolled in the system, and logs their attendance in a CSV file.

## Features

- Real-time face detection and recognition using OpenCV and TensorFlow/Keras models (InceptionV3, ResNet50, DenseNet).
- Streamlit web application for easy interaction with the attendance system.
- Logging attendance records in a CSV file.
- Pickling trained models for easy reusability and deployment.

## Requirements

- Python 3.6+
- Libraries: OpenCV, TensorFlow, Keras, Streamlit, NumPy, Pandas, Matplotlib, PIL (Pillow), Scikit-learn.
- Pre-trained models: InceptionV3, ResNet50, DenseNet (or other models suitable for facial recognition).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Prasadayus/Facial-Attendance-System.git
   cd facial-attendance-system
