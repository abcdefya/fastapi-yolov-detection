import nest_asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import uvicorn
import io
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO

# Apply nest_asyncio for notebook compatibility
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI(
    title="Animal Detection API",
    description="API for detecting animals in images and videos using YOLOv8",
    version="0.2.0"
)

# Define paths
BASE_DIR = Path.cwd()
MODEL_PATH = BASE_DIR / "data" / "models" / "yolov8n.pt"
TEMP_DIR = BASE_DIR / "data" / "temp"

# Create temp directory
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

# COCO ids: 15 = cat, 16 = dog
ANIMAL_CLASSES = [15, 16]

@app.post("/predict/image")
async def predict_image_with_bbox(file: UploadFile = File(...)):
    # Read image file path
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

    # Perform prediction
    results = model(img)

    # bounding boxes drawing
    annotated_img = results[0].plot()

    # Convert image to bytes to return
    is_success, buffer = cv2.imencode(".jpg", annotated_img)
    if not is_success:
        return JSONResponse(content={"error": "Failed to process image"}, status_code=500)

    # Return image as response
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    # Validate file type
    if not (file.content_type and file.content_type.startswith("video/")):
        raise HTTPException(status_code=400, detail="File must be a video (mp4, avi, etc.)")

    # Save video to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=str(TEMP_DIR))
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video file")
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
            # Run YOLO prediction (cat=15, dog=16)
            results = model(frame, classes=ANIMAL_CLASSES)

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


@app.post("/track/video")
async def track_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf: float = 0.25,
    tracker: str = "bytetrack.yaml"  # or "botsort.yaml"
):
    """
    Nhận video, chạy YOLOv8 tracking để gán ID & annotate qua các frame,
    sau đó trả về file MP4 đã annotate.
    """
    # Validate file type
    if not (file.content_type and file.content_type.startswith("video/")):
        raise HTTPException(status_code=400, detail="File must be a video (mp4, avi, etc.)")

    # Save input video temporarily
    in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=str(TEMP_DIR))
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty video file")
        in_tmp.write(data)
        in_tmp.close()

        # Read video information (fps, size) to initialize VideoWriter for output
        probe = cv2.VideoCapture(in_tmp.name)
        if not probe.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open input video")
        fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.release()

        # Temporary file output
        out_path = str(TEMP_DIR / f"tracked_{next(tempfile._get_candidate_names())}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # phổ biến, dễ phát
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise HTTPException(status_code=500, detail="Cannot open VideoWriter for output")

        # Run tracking in stream mode to write annotated file
        # persist=True để giữ ID track giữa các frame
        frames_written = 0
        try:
            stream = model.track(
                source=in_tmp.name,
                stream=True,
                conf=conf,
                classes=ANIMAL_CLASSES,
                tracker=tracker,
                persist=True,
                verbose=False
            )
            for result in stream:
                # result.plot() returns frame BGR with drawn bbox + track id
                annotated = result.plot()
                # Ensure size matches writer
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated = cv2.resize(annotated, (width, height))
                writer.write(annotated)
                frames_written += 1
        finally:
            writer.release()

        if frames_written == 0:
            # No frames written
            os.unlink(out_path) if os.path.exists(out_path) else None
            raise HTTPException(status_code=500, detail="No frames processed during tracking")

        # Return file and schedule deletion after sending
        background_tasks.add_task(os.unlink, out_path)
        background_tasks.add_task(os.unlink, in_tmp.name)
        filename_out = (Path(file.filename).stem if file.filename else "video") + "_tracked.mp4"
        return FileResponse(out_path, media_type="video/mp4", filename=filename_out)

    except Exception as e:
        # Clean up if there is an error
        if os.path.exists(in_tmp.name):
            os.unlink(in_tmp.name)
        raise e
