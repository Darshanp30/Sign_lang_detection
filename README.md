# Sign Language Detection using LSTM

This project implements a real-time sign language recognition system using Long Short-Term Memory (LSTM) networks. The system captures live video from a webcam, extracts hand landmarks using MediaPipe, and classifies the gestures into predefined sign labels using a deep learning model.

## Project Overview

The goal of this project is to bridge communication between hearing-impaired individuals and others by recognizing hand gestures in real time. The system uses a sequence-based approach with temporal modeling to handle the dynamic nature of sign language.

## Features

- Real-time hand gesture recognition using webcam input
- Detection of hand landmarks using MediaPipe
- Custom dataset creation with OpenCV and NumPy
- LSTM-based neural network trained on gesture sequences
- Modular, extensible code structure for adding new gestures

## Methodology

### 1. Data Collection
- Hand landmarks (21 points per hand) are extracted from each video frame using MediaPipe.
- Sequences of 30 consecutive frames are collected per gesture.
- Each frame’s landmarks are saved as 3D coordinates (x, y, z).
- Data is labeled and stored as NumPy arrays for training.

### 2. Model Architecture
- Input Shape: (30, 63) → 30 frames × 21 landmarks × 3 coordinates
- Architecture:
  - LSTM (64 units, return sequences=True)
  - LSTM (128 units)
  - Dense (64), ReLU activation
  - Dense (number of gestures), Softmax activation

### 3. Training
- The model is trained using categorical cross-entropy loss and Adam optimizer.
- Training is performed on the custom-collected dataset with multiple gesture classes.
- Validation accuracy of over 95% was achieved on the limited test set.

### 4. Real-Time Prediction
- Webcam captures a live video stream.
- The last 30 frames are used to build a prediction sequence.
- The trained model predicts the most likely gesture and displays it in real time.

