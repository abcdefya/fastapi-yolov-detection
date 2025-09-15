import nest_asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import io
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO
from app.config.settings import MODEL_PATH, TEMP_DIR

# Apply nest_asyncio for notebook compatibility (optional for production)
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI(
    title="Animal Detection API",
    description="API for detecting animals in images and videos using YOLOv8",
    version="0.1.0"
)

# Load YOLO model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

@app.post("/predict/image")
async def predict_image_with_bbox(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, etc.)")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image")
    
    # Run prediction
    results = model(img, classes=[16, 17])  # Limit to dog (16) and cat (17)
    
    # Draw bounding boxes
    annotated_img = results[0].plot()
    
    # Convert to bytes for response
    is_success, buffer = cv2.imencode(".jpg", annotated_img)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to process image")
    
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video (mp4, avi, etc.)")
    
    # Save video to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=str(TEMP_DIR))
    try:
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Open video
        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot read video")
        
        # Process frames (limit to 100 frames)
        frame_count = 0
        max_frames = 100
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                raise HTTPException(status_code=404, detail="No dogs or cats detected in the first 100 frames")
            
            frame_count += 1
            # Run YOLO prediction
            results = model(frame, classes=[16, 17])  # Limit to dog (16) and cat (17)
            
            # Check for detected objects
            if len(results[0].boxes) > 0:
                annotated_frame = results[0].plot()
                
                # Convert to bytes for response
                is_success, buffer = cv2.imencode(".jpg", annotated_frame)
                if not is_success:
                    raise HTTPException(status_code=500, detail="Failed to process frame")
                
                cap.release()
                return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
        
        cap.release()
        raise HTTPException(status_code=404, detail="No dogs or cats detected in the first 100 frames")
    
    finally:
        # Clean up temporary file
        if temp_file.name and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)