Here’s a 3-lane vehicle counting Python project.
Multi-Lane Vehicle Counting using Python & Computer Vision in this project i have stored the youtube video directly in project folder which was fetching from folder as well. 
Project Overview

This project detects, tracks, and counts vehicles in **three distinct lanes** from a video feed using computer vision techniques. It maintains lane-wise counts and supports real-time processing.

Features
* Tracks vehicles across frames using **DeepSORT**.
* Lane-wise vehicle counting with virtual entry/exit lines.
* Handles occlusion and partial visibility.
* Optimized for **real-time performance**.

Requirements
* Python 3.12
* OpenCV
* NumPy
* TensorFlow / PyTorch (depending on model used)
* DeepSORT or equivalent tracking library
* Optional: GPU support for faster detection

Install dependencies using pip:
pip install opencv-python numpy matplotlib torch torchvision
pip install scikit-learn filterpy

Setup Instructions
Download Pre-trained Model

Challenges & Solutions
Occlusion: Used DeepSORT tracker with re-identification.
* Lane Misclassification: Applied perspective transformation and ROI calibration.
* Double Counting: Implemented crossing-line logic with ID state flags.
* real-time Processing: Resized frames, used GPU acceleration, and hybrid detection-tracking approach.

Output
* Video with bounding boxes around vehicles.
* Lane-wise vehicle counts displayed on video frames.
* CSV or JSON log of counts per lane.

Future Improvements
* Integrate speed estimation per vehicle.
* Deploy on edge devices.
* Add multi-camera synchronization for larger traffic monitoring.

Future Improvements
* Speed estimation per vehicle.
* Edge device deployment (Jetson Nano / Raspberry Pi).
* Multi-camera synchronized traffic analysis.
