# Project_Deep_Blue
# Problem Statement: Estimating Physical Attributes and Object Dimensions from Selfies
This repository contains a Flask-based web application that estimates physical attributes and object dimensions from selfies. The application provides predictions for age, gender, height,weight, bmi for humans detected in image and further estimates object dimensions for the objects detected.
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

# Technologies Used
-Backend: Flask (Python)
Frontend: HTML, CSS, JavaScript
Machine Learning Models:
Caffe (Age & Gender Prediction)
PyTorch (Height, Weight, BMI Estimation)
YOLOv8n (Object Detection)
