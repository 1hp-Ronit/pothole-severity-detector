from ultralytics import YOLO

model = YOLO("model/best.pt")

def classify_severity(x1, y1, x2, y2, img_w, img_h):
    box_area = (x2 - x1) * (y2 - y1)
    ratio = box_area / (img_w * img_h)
    if ratio < 0.015:
        return "minor"
    elif ratio < 0.06:
        return "moderate"
    return "severe"

def run_inference(image_path, conf=0.35, iou=0.4):
    results = model(image_path, conf=conf, iou=iou)[0]
    img_h, img_w = results.orig_shape
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        severity = classify_severity(x1, y1, x2, y2, img_w, img_h)
        detections.append({
            "bbox": [round(x1), round(y1), round(x2), round(y2)],
            "confidence": round(confidence, 3),
            "severity": severity
        })
    return detections, results