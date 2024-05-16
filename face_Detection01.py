import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image

# Function to perform speech recognition
def recognize_speech():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Capture audio from microphone
    with sr.Microphone() as source:
        print("Please say your name:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Use recognizer to convert audio to text
        name = recognizer.recognize_google(audio)
        return name
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return None

# Function to perform speech synthesis
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Load the pre-trained face recognition model
# Assuming you have a PyTorch model trained on GPU
model = torch.load("face_recognition_model.pth")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize lists to store known face encodings and names
try:
    with open("known_faces.pkl", "rb") as file:
        known_face_encodings, known_face_names = pickle.load(file)
except FileNotFoundError:
    known_face_encodings = []
    known_face_names = []

# Initialize unknown face frames
unknown_face_frames = []

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert frame to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize and transform frame for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(frame_pil).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Convert output to probabilities and class index
    probabilities = torch.softmax(outputs, dim=1)[0]
    predicted_class_index = torch.argmax(probabilities).item()

    # Get predicted class label
    if predicted_class_index < len(known_face_names):
        name = known_face_names[predicted_class_index]
    else:
        name = "Unknown"

    # Display the results
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (10, 30), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
