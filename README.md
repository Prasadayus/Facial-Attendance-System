# Facial Attendance System

Facial Attendance System is a project aimed at automating attendance management using facial recognition. This system detects faces in real-time using a webcam, recognizes the faces of individuals enrolled in the system, and logs their attendance in a CSV file.

## Features

- Real-time face detection and recognition using OpenCV and TensorFlow/Keras models (InceptionV3, ResNet50, DenseNet).
- Utilized CNN model, transfer learning with ResNet-50, InceptionV3, and DenseNet-121 to achieve accurate facial recognition.
- Selected DenseNet-121 model for capturing due to its highest accuracy of 96%, increasing accurcy from  84%  achieved by the CNN model.
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
     ```
2.Install dependencies:
   ```
      pip install -r requirements.txt
   ```

## Usage
### Capturing Images for Training
To capture images for training the facial recognition models:

1.Run the image capture script:
 ```
python capture_images.py
 ```

2.Follow the instructions to enter the client's name and capture images.

## Model Training
To train the facial recognition models:

1.Ensure captured images are stored in the imgs/ directory (not provided in the repository due to privacy reasons).

2.Train the model using TensorFlow/Keras:
```
python train_model.py
```

Select the appropriate model architecture and follow the training process.

## Logging Attendance
To log attendance using the Streamlit web application:

1.Launch the Streamlit web application:
```
streamlit run app.py
```
2.Use the web interface to start the webcam, recognize faces, and log attendance.

3.Attendance records are stored in CSV files located in the data/ directory.

## Contributing
Contributions are welcome! If you have any suggestions, feature requests, or bug reports, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
