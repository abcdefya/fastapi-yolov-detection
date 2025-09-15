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
import logging
from app.config.config import MODEL_PATH, TEMP_DIR, BASE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Apply nest_asyncio for notebook compatibility (optional for production)
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI(
    title="Animal Detection API",
    description="API for detecting animals in images and videos using YOLOv8",
    version="0.1.0"
)

# Log application startup
logger.info("Starting Animal Detection API")

# Load YOLO model
if not MODEL_PATH.exists():
    logger.error(f"Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
logger.info(f"YOLOv8 model loaded from {MODEL_PATH}")

@app.post("/predict/image")
async def predict_image_with_bbox(file: UploadFile = File(...)):
    # Đọc file ảnh
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Chạy prediction
    results = model(img)
    
    # Log detected objects to see if anything is found
    detections = results[0].boxes
    if len(detections) > 0:
        logger.info(f"Found {len(detections)} objects.")
        for box in detections:
            class_name = results[0].names[int(box.cls)]
            confidence = box.conf.item()
            logger.info(f"  - Detected {class_name} with confidence {confidence:.2f}")
    else:
        logger.info("No objects detected in the image.")
    
    # Vẽ bounding boxes
    annotated_img = results[0].plot()
    
    # Chuyển ảnh thành bytes để trả về
    is_success, buffer = cv2.imencode(".jpg", annotated_img)
    if not is_success:
        return JSONResponse(content={"error": "Failed to process image"}, status_code=500)
    
    # Trả về ảnh dưới dạng response
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    logger.info(f"Received video upload: {file.filename}")
    
    # Validate file type
    if not file.content_type.startswith("video/"):
        logger.error(f"Invalid file type for video: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be a video (mp4, avi, etc.)")
    
    # Save video to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=str(TEMP_DIR))
    try:
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()
        logger.info(f"Saved temporary video file: {temp_file.name}")
        
        # Open video
        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            logger.error(f"Cannot read video: {temp_file.name}")
            raise HTTPException(status_code=400, detail="Cannot read video")
        
        # Process frames (limit to 100 frames)
        frame_count = 0
        max_frames = 100
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                logger.warning("End of video or no frames available")
                raise HTTPException(status_code=404, detail="No dogs or cats detected in the first 100 frames")
            
            frame_count += 1
            # Run YOLO prediction
            logger.debug(f"Processing frame {frame_count}")
            results = model(frame, classes=[16, 17])  # Limit to dog (16) and cat (17)
            
            # Check for detected objects
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_name = results[0].names[int(box.cls)]
                    confidence = box.conf.item()
                    logger.info(f"Detected {class_name} with confidence {confidence:.2f} in frame {frame_count}")
                
                annotated_frame = results[0].plot()
                
                # Convert to bytes for response
                is_success, buffer = cv2.imencode(".jpg", annotated_frame)
                if not is_success:
                    logger.error("Failed to encode frame")
                    raise HTTPException(status_code=500, detail="Failed to process frame")
                
                logger.info("Video frame processed successfully")
                cap.release()
                return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
        
        cap.release()
        logger.warning("No dogs or cats detected in the first 100 frames")
        raise HTTPException(status_code=404, detail="No dogs or cats detected in the first 100 frames")
    
    finally:
        # Clean up temporary file
        if temp_file.name and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
            logger.info(f"Deleted temporary video file: {temp_file.name}")

if __name__ == "__main__":
    logger.info("Starting Uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)