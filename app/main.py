from fastapi import FastAPI, UploadFile, File
import shutil, os
from app.inference import run_inference

app = FastAPI(title="Pothole Severity Detector")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections = run_inference(temp_path)
    os.remove(temp_path)

    return {"filename": file.filename, "detections": detections}