import torch
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO  # Import YOLOv8

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load Age & Gender Models (Caffe)
AGE_MODEL = "models/age_net.caffemodel"
AGE_PROTO = "models/age_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

GENDER_BUCKETS = ["Male", "Female"]

# Load PyTorch Model for Height, Weight, BMI Prediction
class HeightWeightBMIModel(torch.nn.Module):
    def __init__(self):
        super(HeightWeightBMIModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load trained PyTorch model
checkpoint = torch.load("data_new_model.pth")
model = HeightWeightBMIModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load scaler from training
scaler_y = checkpoint["scaler_y"]

# Detect Faces (MediaPipe)
def detect_faces(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    faces = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            faces.append((x, y, w, h))

    return faces

# Detect Age & Gender (Returns exact age instead of buckets)
def detect_age_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.896), swapRB=False)

    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_preds[0].argmax()  # 0: Male, 1: Female

    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    
    # Age bucket values (midpoints for better approximation)
    age_buckets = np.array([5, 18, 20, 21, 22, 24, 36, 65])

    # Compute estimated age as a weighted sum
    estimated_age = int(round(np.dot(age_preds[0], age_buckets)))
    estimated_age = max(1, min(estimated_age, 100))  # Ensure it's within range

    return estimated_age, gender

# Estimate Height, Weight, BMI, risk
def health_risk(bmi, age):
    if age < 10:
        bmi_ranges = [14, 16, 18, 20, 22, 24, 26, 30, 35]
    elif 10 <= age < 18:
        bmi_ranges = [15.5, 17, 19.5, 22, 24, 26, 28, 32, 37]
    elif 18 <= age < 27:
        bmi_ranges = [16, 18.5, 21.9, 24.9, 26.9, 29.9, 34.9, 39.9]
    elif 27 <= age < 40:
        bmi_ranges = [17, 19, 23, 25, 27, 30, 35, 40]
    else:  # Above 40
        bmi_ranges = [18, 20, 24, 26, 28, 32, 36, 41]

    if bmi < bmi_ranges[0]:
        return "Severely Underweight - Immediate nutritional support needed. Consult a healthcare provider."
    elif bmi < bmi_ranges[1]:
        return "Underweight - Consider a balanced diet to gain healthy weight."
    elif bmi < bmi_ranges[2]:
        return "Healthy Weight (Lower Range) - Good health, ensure proper nutrition for long-term well-being."
    elif bmi < bmi_ranges[3]:
        return "Healthy Weight (Upper Range) - Maintain a balanced lifestyle with regular exercise."
    elif bmi < bmi_ranges[4]:
        return "Mild Overweight - Start incorporating more physical activity and mindful eating habits."
    elif bmi < bmi_ranges[5]:
        return "Overweight - Increased risk of health issues. Prioritize a nutritious diet and active routine."
    elif bmi < bmi_ranges[6]:
        return "Obesity (Class 1) - Lifestyle modifications are necessary to prevent chronic health conditions."
    elif bmi < bmi_ranges[7]:
        return "Obesity (Class 2) - High health risk. Seek professional guidance for weight management."
    else:
        return "Severe Obesity (Class 3) - Urgent intervention required. Medical supervision recommended."

    
def estimate_height_weight_bmi(age, gender):
    input_data = np.array([[age, gender]], dtype=np.float32)
    input_tensor = torch.tensor(input_data)

    with torch.no_grad():
        predicted_scaled = model(input_tensor).numpy()

    predicted_values = scaler_y.inverse_transform(predicted_scaled)

    height = round(float(predicted_values[0][0]), 2)
    weight = round(float(predicted_values[0][1]), 2)
    bmi = round(float(predicted_values[0][2]),1)

    if gender == 1:  # Female
        height -= 4
        weight -= 4
        height = round(height, 2)
        weight = round(weight, 2)

    risk = health_risk(bmi, age)

    return height, weight, bmi, risk

# COCO dimensions (width, height in centimeters as strings)
coco_dimensions = {
    "bicycle": (160.0, 180.0, 50.0, 70.0),
    "car": (390.0, 410.0, 170.0, 190.0),
    "motorcycle": (200.0, 220.0, 70.0, 90.0),
    "airplane": (3600.0, 3800.0, 3500.0, 3700.0),
    "bus": (1190.0, 1210.0, 240.0, 260.0),
    "train": (23900.0, 24100.0, 290.0, 310.0),
    "truck": (690.0, 710.0, 240.0, 260.0),
    "boat": (490.0, 510.0, 190.0, 210.0),
    "traffic light": (29.0, 31.0, 9.0, 11.0),
    "fire hydrant": (38.0, 42.0, 38.0, 42.0),
    "stop sign": (73.0, 77.0, 73.0, 77.0),
    "parking meter": (29.0, 31.0, 9.0, 11.0),
    "bench": (140.0, 160.0, 45.0, 55.0),
    "bird": (24.0, 26.0, 9.0, 11.0),
    "cat": (48.0, 52.0, 18.0, 22.0),
    "dog": (73.0, 77.0, 28.0, 32.0),
    "horse": (190.0, 210.0, 75.0, 85.0),
    "sheep": (140.0, 160.0, 65.0, 75.0),
    "cow": (240.0, 260.0, 95.0, 105.0),
    "elephant": (590.0, 610.0, 290.0, 310.0),
    "bear": (240.0, 260.0, 110.0, 130.0),
    "zebra": (240.0, 260.0, 90.0, 110.0),
    "giraffe": (390.0, 410.0, 110.0, 130.0),
    "backpack": (43.0, 47.0, 28.0, 32.0),
    "umbrella": (88.0, 92.0, 8.0, 12.0),
    "handbag": (33.0, 37.0, 23.0, 27.0),
    "tie": (140.0, 160.0, 7.0, 9.0),
    "suitcase": (53.0, 57.0, 33.0, 37.0),
    "frisbee": (25.0, 29.0, 25.0, 29.0),
    "skis": (170.0, 190.0, 8.0, 12.0),
    "snowboard": (140.0, 160.0, 28.0, 32.0),
    "sports ball": (20.0, 24.0, 20.0, 24.0),
    "kite": (110.0, 130.0, 110.0, 130.0),
    "baseball bat": (80.0, 90.0, 6.0, 8.0),
    "baseball glove": (28.0, 32.0, 18.0, 22.0),
    "skateboard": (75.0, 85.0, 18.0, 22.0),
    "surfboard": (210.0, 230.0, 55.0, 65.0),
    "tennis racket": (65.0, 75.0, 28.0, 32.0),
    "bottle": (23.0, 27.0, 6.0, 8.0),
    "wine glass": (18.0, 22.0, 7.0, 9.0),
    "cup": (9.0, 11.0, 7.0, 9.0),
    "fork": (18.0, 22.0, 1.5, 2.5),
    "knife": (23.0, 27.0, 1.5, 2.5),
    "spoon": (18.0, 22.0, 1.5, 2.5),
    "bowl": (14.0, 16.0, 14.0, 16.0),
    "banana": (18.0, 22.0, 3.5, 4.5),
    "apple": (7.0, 9.0, 7.0, 9.0),
    "sandwich": (11.0, 13.0, 5.0, 7.0),
    "orange": (7.0, 9.0, 7.0, 9.0),
    "broccoli": (14.0, 16.0, 9.0, 11.0),
    "carrot": (18.0, 22.0, 2.5, 3.5),
    "hot dog": (14.0, 16.0, 4.5, 5.5),
    "pizza": (28.0, 32.0, 28.0, 32.0),
    "donut": (9.0, 11.0, 9.0, 11.0),
    "cake": (18.0, 22.0, 18.0, 22.0),
    "chair": (48.0, 52.0, 48.0, 52.0),
    "couch": (190.0, 210.0, 85.0, 95.0),
    "potted plant": (28.0, 32.0, 28.0, 32.0),
    "bed": (190.0, 210.0, 150.0, 170.0),
    "dining table": (170.0, 190.0, 85.0, 95.0),
    "tv": (98.0, 102.0, 58.0, 62.0),
    "laptop": (33.0, 37.0, 22.0, 26.0),
    "mouse": (11.0, 13.0, 5.5, 6.5),
    "remote": (17.0, 19.0, 4.0, 5.0),
    "keyboard": (43.0, 47.0, 14.0, 16.0),
    "cell phone": (15.0, 17.0, 7.0, 8.0),
    "microwave": (48.0, 52.0, 38.0, 42.0),
    "oven": (73.0, 77.0, 58.0, 62.0),
    "toaster": (28.0, 32.0, 18.0, 22.0),
    "sink": (58.0, 62.0, 48.0, 52.0),
    "refrigerator": (68.0, 72.0, 68.0, 72.0),
    "book": (24.0, 26.0, 17.0, 19.0),
    "clock": (28.0, 32.0, 28.0, 32.0),
    "vase": (24.0, 26.0, 9.0, 11.0),
    "scissors": (14.0, 16.0, 4.5, 5.5),
    "teddy bear": (38.0, 42.0, 28.0, 32.0),
    "hair drier": (18.0, 22.0, 9.0, 11.0),
    "toothbrush": (17.0, 19.0, 1.5, 2.5)
}



# Object Detection with YOLOv8
def detect_objects(image_path):
    model = YOLO("yolov8n.pt")  # Load YOLOv8 model
    image = cv2.imread(image_path)
    results = model(image)[0]
    
    detected_objects = []
    PIXELS_PER_CM = 45  # Conversion factor (can vary depending on camera specs)
    
    for result in results.boxes.data:
        x1, y1, x2, y2, conf, class_id = result.tolist()
        label = model.names[int(class_id)].strip().lower()  # Convert label to lowercase and strip any extra spaces
        confidence = round(float(conf), 2)

        # Debugging: Print the detected label
        print(f"Detected label: '{label}'")

        # Ignore "person"
        if label == "person":
            continue

        # Check if label exists in coco_dimensions dictionary
        if label in coco_dimensions:
            length_upper, length_lower, width_upper, width_lower = coco_dimensions[label]  # Get length and width ranges
            import random

            # Generate random values close to the average within the range
            avg_length = (length_upper + length_lower) / 2.0
            avg_width = (width_upper + width_lower) / 2.0
            
            random_length = round(random.uniform(avg_length - 1.25, avg_length + 1.25), 2)
            random_width = round(random.uniform(avg_width - 1.25, avg_width + 1.25), 2)

            
            # Debugging: Print the dimensions fetched
            print(f"Dimensions for {label}: {random_length:.1f} cm x {random_width:.1f} cm")
            
            detected_objects.append({
                "label": label,
                "length": random_length,
                "width": random_width
            })
        else:
            print(f"Dimensions for {label} not found in coco_dimensions.")  # Debugging message for missing labels
            detected_objects.append({
                "label": label,
                "length": "Unknown",
                "width": "Unknown"
            })
    
    return detected_objects


# Process Camera Input
def process_camera():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
