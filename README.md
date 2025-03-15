# Problem Statement: Estimating Physical Attributes and Object Dimensions from Selfies
This repository contains a Flask-based web application that estimates physical attributes and object dimensions from selfies. The application predicts age, gender, height, weight, and BMI for humans detected in an image and further estimates object dimensions for detected objects. Additionally, based on the predicted age and BMI, the application provides simple and concise health recommendations.
## Technologies Used
- **Backend**:
  - Flask (Python)
- **Frontend**:
  - HTML  
  - CSS  
  - JavaScript  
- **Machine Learning Models**:
  - Caffe (Age & Gender Prediction)  
  - PyTorch (Height, Weight, BMI Estimation)  
  - YOLOv8n (Object Detection)  
## Steps to Run the Application

1. Extract the ZIP File
   ```bash
   unzip Project_Deep_Blue.zip
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
## Usage
1. Choose an image from your device or capture a live photo using your webcam.
2. Click on the Analyze button to process the image.
3. The application will display estimated age, gender, height, weight, BMI, and detected objects.


