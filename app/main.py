from fastapi import FastAPI, UploadFile, File, Response
from fastapi.staticfiles import StaticFiles
import shutil, os, cv2
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
    detections, _ = run_inference(temp_path)
    os.remove(temp_path)
    return {"filename": file.filename, "detections": detections}

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    _, results = run_inference(temp_path)
    annotated = results.plot()
    _, encoded = cv2.imencode(".jpg", annotated)
    os.remove(temp_path)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")

app.mount("/ui", StaticFiles(directory="app/static", html=True), name="static")