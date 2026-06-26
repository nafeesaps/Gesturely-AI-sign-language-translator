Gesturely
Gesturely is an AI-powered real-time sign language translation system that helps improve communication between deaf or hard-of-hearing individuals and non-signers. The system uses computer vision and deep learning techniques to recognize hand gestures and convert them into readable text in real time.

Gesturely supports both alphabet-level and word-level sign recognition using MediaPipe hand tracking and LSTM neural networks.

Features
Real-time sign language recognition
Sign-to-Text translation
Text-to-Sign visualization
Letter Mode for alphabet recognition
Word Mode for dynamic gesture prediction
MediaPipe hand landmark detection
LSTM-based gesture classification
Flask-based web application
Lightweight and accessible solution using standard webcams
Tech Stack
Python
Flask
OpenCV
MediaPipe
TensorFlow / Keras
NumPy
HTML
CSS
JavaScript
Installation
Clone the repository:

git clone https://github.com/aleesha2812/gesturely.git
cd gesturely
Install dependencies:

pip install -r requirements.txt
Usage
Run the application:

python app.py
Open your browser and visit:

http://127.0.0.1:5000
Make sure your webcam is enabled.

How It Works
Captures live webcam input
Detects 21 hand landmarks using MediaPipe
Extracts landmark coordinates as feature vectors
Uses LSTM models for gesture recognition
Displays recognized text output in real time
Converts typed text into sign visualizations
