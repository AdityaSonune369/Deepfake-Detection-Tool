from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
from detector import DeepfakeDetector
from video_processor import VideoProcessor

app = FastAPI(title="Deepfake Detection Tool")

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Initialize detector
detector = DeepfakeDetector(use_cuda=True)
video_processor = VideoProcessor(detector)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    # Basic validation
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = f"temp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        result = detector.predict(temp_path)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
        
    temp_path = f"temp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Process every 30th frame (approx 1 sec for 30fps)
        result = video_processor.process_video(temp_path, frame_interval=30)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
