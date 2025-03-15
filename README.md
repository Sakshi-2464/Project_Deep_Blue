# Problem Statement: Estimating Physical Attributes and Object Dimensions from Selfies
This repository contains a Flask-based web application that estimates physical attributes and object dimensions from selfies. The application provides predictions for age, gender, height,weight, bmi for humans detected in image and further estimates object dimensions for the objects detected.
Age & Gender Prediction: Utilizes Caffe models to estimate a person’s age and gender from selfies.

Height, Weight & BMI Estimation: Uses a PyTorch model to predict physical attributes based on images.

Object Detection: Implements YOLOv8n trained on the COCO dataset for detecting objects in the images.

User Options: Users can either upload an image or use their camera to capture a live photo.

Web Interface: Built using Flask, HTML, CSS, and JavaScript for an interactive user experience.

Image Storage: Uploaded and captured images are saved for further processing.
# Steps to Run the Application

1. Extract the ZIP File
   ```bash
   unzip data.zip
   ```
2. Navigate to the Project Directory
   ```bash 
   cd Project_Deep_Blue
   ```
3. Install Required Dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask Application
   ```bash
   python app.py
   ```
5. Access the Web Application

   Open your browser and visit:
http://127.0.0.1:5000
# Usage
1. Choose an image from your device or capture a live photo using your webcam.
2. Click on the Analyze button to process the image.
3. The application will display estimated age, gender, height, weight, BMI, and detected objects.
# Technologies Used
-Backend: Flask (Python)
Frontend: HTML, CSS, JavaScript
Machine Learning Models:
Caffe (Age & Gender Prediction)
PyTorch (Height, Weight, BMI Estimation)
YOLOv8n (Object Detection)
