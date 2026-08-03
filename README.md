# Pothole Severity Detector

Detects potholes in road images and classifies their severity (minor/moderate/severe) using a fine-tuned YOLOv8 model, served via a FastAPI endpoint.

## Problem
Manual road damage reporting in Indian cities is slow and inconsistent. This project provides an automated way to detect potholes from a photo and estimate severity, enabling faster, crowdsourced road-quality reporting.

## How it works
1. YOLOv8n fine-tuned on a pothole detection dataset (1.2k+ images)
2. Detected bounding box area (relative to image size) is used as a proxy for severity
3. Exposed via a FastAPI `/detect` endpoint — upload an image, get back bounding boxes + severity per detection

## Results
- mAP50: 0.60 | Precision: 0.62 | Recall: 0.59
- Trained for 50 epochs on Colab (T4 GPU), ~13 min

## Tech stack
YOLOv8 (Ultralytics) · FastAPI · Docker · Python 3.11

## Run locally
\`\`\`bash
pip install -r requirements.txt
uvicorn app.main:app --reload
\`\`\`
Visit `http://127.0.0.1:8000/docs` to test.

## Live demo
[link coming soon]

## Dataset
[Pothole Detection using YOLOv8](https://universe.roboflow.com/parul-university-wsshr/pothole-detection-using-yolov8) — Parul University, via Roboflow Universe