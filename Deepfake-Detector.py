import sys
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import time
import numpy as np
from PIL import Image, ImageGrab
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
import pygetwindow as gw

# Function to preprocess a single frame
def preprocess_frame(frame, transform):
    pil_image = Image.fromarray(frame)  # Converts to PIL Image
    pil_image = transform(pil_image)  # Apply transformations
    return pil_image

# Loads the pretrained model
model_path = 'deepfake_detection_model_epoch7.pth'
model = timm.create_model('efficientnet_b0', pretrained=False)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Defines transformations for video frames
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Video capture status
capture_active = False

# VideoAnalyzer class for video analysis window
class VideoAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Analyzer")

        # Label to display the analysis result
        self.result_label = QLabel(self)
        self.result_label.setGeometry(100, 50, 600, 50)
        self.result_label.setText("")
        self.result_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Creates a button to start/stop video capture
        self.toggle_button = QPushButton("Start Capture", self)
        self.toggle_button.setGeometry(150, 520, 150, 50)
        self.toggle_button.clicked.connect(self.toggle_capture)

        # Creates a timer to update the video processing
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(100)

        # Sets the window flags to keep the window on top and transparent
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Gets the screen dimensions
        self.update_screen_geometry()

        # Flag to track if the current prediction is "Real"
        self.is_real_prediction = False
        self.real_prediction_start_time = None
        self.display_duration = 4500

    def update_screen_geometry(self):
        try:
            screen = gw.getWindowsWithTitle('Desktop')[0]
            self.setGeometry(screen.left, screen.top, screen.width, screen.height)
        except IndexError:
            # Handle case where no window with the title 'Desktop' is found
            self.setGeometry(0, 0, 1920, 1080)

    def toggle_capture(self):
        global capture_active
        capture_active = not capture_active
        if capture_active:
            self.result_label.setStyleSheet("color: red;")
            self.result_label.setText("Video classification result: Deepfake")
            self.toggle_button.setText("Stop Capture")
        else:
            self.result_label.setText("")
            self.toggle_button.setText("Start Capture")

    def process_frame(self):
        global capture_active

        if capture_active:
            # Capture screen frame
            screen = ImageGrab.grab(bbox=(self.geometry().left(), self.geometry().top(), self.geometry().right(), self.geometry().bottom()))
            frame = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)  # Convert PIL Image to OpenCV format

            # Preprocess frame
            input_tensor = preprocess_frame(frame, transform)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

            # Model inference
            with torch.no_grad():
                output = model(input_tensor)

            # Perform analysis based on model output
            _, predicted = torch.max(output, 1)
            is_real = predicted.item() < 1

            # Update label based on prediction
            if is_real:
                if not self.is_real_prediction:
                    self.result_label.setStyleSheet("color: green;")
                    self.result_label.setText("Video classification result: Real")
                    self.is_real_prediction = True
                    self.real_prediction_start_time = time.time()
                    self.deepfake_display_start_time = None
            else:
                if self.is_real_prediction:
                    if self.real_prediction_start_time is not None:
                        current_time = time.time()
                        if current_time - self.real_prediction_start_time >= self.display_duration / 1000:  # Convert to seconds
                            self.result_label.setStyleSheet("color: red;")
                            self.result_label.setText("Video classification result: Deepfake")
                            self.is_real_prediction = False
                            self.real_prediction_start_time = None

# DeepfakeWindow class for the main window with start and quit buttons
class DeepfakeWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Deepfakedetectinator")
        self.setGeometry(100, 100, 550, 350)
        self.setStyleSheet("background-color: black; color: green;")

        # Create a heading label
        self.heading_label = QLabel("DEEPFAKEDETECTINATOR", self)
        self.heading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heading_label.setGeometry(50, 20, 450, 150)
        self.heading_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.heading_label.setStyleSheet("color: green;")

        # Create buttons
        self.start_button = QPushButton("Start Analysis", self)
        self.start_button.setGeometry(125, 150, 150, 50)
        self.start_button.clicked.connect(self.start_analysis)

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.setGeometry(275, 150, 150, 50)
        self.quit_button.clicked.connect(self.quit_program)

    def start_analysis(self):
        # Create and show the VideoAnalyzer window
        self.analyzer_window = VideoAnalyzer()
        self.analyzer_window.show()

    def quit_program(self):
        QApplication.quit()

def main():
    app = QApplication(sys.argv)

    # Create and show the DeepfakeWindow
    deepfake_window = DeepfakeWindow()
    deepfake_window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
