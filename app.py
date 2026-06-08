"""
RoadPulse API - FastAPI Backend for Road Damage Detection
Supports 3 modules: Road Damage Detection, Privacy Blur, Combined Processing
"""

import os
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from enum import Enum

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI(
    title="RoadPulse API",
    description="Edge-based Road Safety Network - Road Damage Detection & Privacy Protection",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model paths
ROAD_DAMAGE_MODEL_PATH = MODELS_DIR / "YOLOv8_Small_RDD.pt"
FACE_MODEL_PATH = MODELS_DIR / "yolov8n-face.pt"
PLATE_MODEL_PATH = MODELS_DIR / "license_plate_detector.pt"

# Class labels for road damage
ROAD_DAMAGE_CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

# Enums
class ProcessingMode(str, Enum):
    ROAD_DAMAGE = "road_damage"
    PRIVACY_BLUR = "privacy_blur"
    COMBINED = "combined"

class BlurMethod(str, Enum):
    GAUSSIAN = "gaussian"
    PIXELATE = "pixelate"

# Pydantic models
class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float
    box: List[int]

class ProcessingResult(BaseModel):
    success: bool
    message: str
    detections: Optional[List[Detection]] = None
    output_url: Optional[str] = None
    stats: Optional[dict] = None

# Global model storage
models = {}

def load_models():
    """Load YOLO models"""
    global models

    if ROAD_DAMAGE_MODEL_PATH.exists():
        models["road_damage"] = YOLO(str(ROAD_DAMAGE_MODEL_PATH))
        print(f"Loaded road damage model from {ROAD_DAMAGE_MODEL_PATH}")

    if FACE_MODEL_PATH.exists():
        models["face"] = YOLO(str(FACE_MODEL_PATH))
        print(f"Loaded face detection model from {FACE_MODEL_PATH}")

    if PLATE_MODEL_PATH.exists():
        models["plate"] = YOLO(str(PLATE_MODEL_PATH))
        print(f"Loaded license plate model from {PLATE_MODEL_PATH}")

# Blur operations
def gaussian_blur_region(frame, x1, y1, x2, y2, strength=8.0):
    """Apply Gaussian blur to a region"""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    kw = max(3, int((x2-x1)/strength) | 1)
    kh = max(3, int((y2-y1)/strength) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kw, kh), 0)

def pixelate_region(frame, x1, y1, x2, y2, blocks=12):
    """Apply pixelation to a region"""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    fx = max(1, w // blocks)
    fy = max(1, h // blocks)
    small = cv2.resize(roi, (max(1, w//fx), max(1, h//fy)), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def expand_box(x1, y1, x2, y2, scale, W, H):
    """Expand bounding box by scale factor"""
    w, h = x2-x1, y2-y1
    cx, cy = x1 + w/2, y1 + h/2
    w2, h2 = w*scale, h*scale
    nx1 = int(max(0, cx - w2/2))
    ny1 = int(max(0, cy - h2/2))
    nx2 = int(min(W-1, cx + w2/2))
    ny2 = int(min(H-1, cy + h2/2))
    return nx1, ny1, nx2, ny2

def detect_road_damage(frame, conf_threshold=0.5):
    """Detect road damage in frame"""
    if "road_damage" not in models:
        return [], frame

    h_ori, w_ori = frame.shape[:2]
    frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

    results = models["road_damage"].predict(frame_resized, conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].astype(int)

                # Scale back to original size
                scale_x = w_ori / 640
                scale_y = h_ori / 640
                x1 = int(xyxy[0] * scale_x)
                y1 = int(xyxy[1] * scale_y)
                x2 = int(xyxy[2] * scale_x)
                y2 = int(xyxy[3] * scale_y)

                detections.append(Detection(
                    class_id=class_id,
                    label=ROAD_DAMAGE_CLASSES[class_id],
                    confidence=round(conf, 3),
                    box=[x1, y1, x2, y2]
                ))

    # Draw detections on frame
    annotated = results[0].plot() if results else frame_resized
    annotated = cv2.resize(annotated, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    return detections, annotated

def detect_and_blur_privacy(frame, blur_faces=True, blur_plates=True,
                           blur_method="gaussian", conf_threshold=0.35, scale=1.35):
    """Detect and blur faces/license plates"""
    H, W = frame.shape[:2]
    all_boxes = []

    if blur_faces and "face" in models:
        results = models["face"](frame, conf=conf_threshold, verbose=False)
        if results and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                all_boxes.append(tuple(map(int, box)))

    if blur_plates and "plate" in models:
        results = models["plate"](frame, conf=conf_threshold, verbose=False)
        if results and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                all_boxes.append(tuple(map(int, box)))

    # Apply blur to each detected region
    for (x1, y1, x2, y2) in all_boxes:
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, scale, W, H)
        if blur_method == "gaussian":
            gaussian_blur_region(frame, x1, y1, x2, y2)
        else:
            pixelate_region(frame, x1, y1, x2, y2)

    return len(all_boxes), frame

# API Endpoints

@app.get("/")
async def root():
    """API Health Check"""
    return {
        "status": "online",
        "app": "RoadPulse API",
        "version": "1.0.0",
        "modules": {
            "road_damage": "road_damage" in models,
            "face_detection": "face" in models,
            "plate_detection": "plate" in models
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy"}

@app.post("/api/process/image", response_model=ProcessingResult)
async def process_image(
    file: UploadFile = File(...),
    mode: ProcessingMode = Form(ProcessingMode.ROAD_DAMAGE),
    confidence: float = Form(0.5),
    blur_faces: bool = Form(True),
    blur_plates: bool = Form(True),
    blur_method: BlurMethod = Form(BlurMethod.GAUSSIAN)
):
    """Process a single image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file_id}_input.jpg"
    output_path = OUTPUT_DIR / f"{file_id}_output.jpg"

    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(400, "Could not decode image")

        detections = []
        stats = {"faces_blurred": 0, "plates_blurred": 0, "damages_detected": 0}

        if mode == ProcessingMode.ROAD_DAMAGE:
            detections, annotated = detect_road_damage(frame, confidence)
            stats["damages_detected"] = len(detections)
            frame = annotated

        elif mode == ProcessingMode.PRIVACY_BLUR:
            count, frame = detect_and_blur_privacy(
                frame, blur_faces, blur_plates, blur_method.value, confidence
            )
            stats["faces_blurred"] = count

        elif mode == ProcessingMode.COMBINED:
            # First blur privacy
            count, frame = detect_and_blur_privacy(
                frame, blur_faces, blur_plates, blur_method.value, 0.35
            )
            stats["faces_blurred"] = count

            # Then detect road damage
            detections, annotated = detect_road_damage(frame, confidence)
            stats["damages_detected"] = len(detections)
            frame = annotated

        # Save output
        cv2.imwrite(str(output_path), frame)

        return ProcessingResult(
            success=True,
            message=f"Processed with {mode.value} mode",
            detections=detections if detections else None,
            output_url=f"/outputs/{file_id}_output.jpg",
            stats=stats
        )

    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/process/video", response_model=ProcessingResult)
async def process_video(
    file: UploadFile = File(...),
    mode: ProcessingMode = Form(ProcessingMode.ROAD_DAMAGE),
    confidence: float = Form(0.5),
    blur_faces: bool = Form(True),
    blur_plates: bool = Form(True),
    blur_method: BlurMethod = Form(BlurMethod.GAUSSIAN)
):
    """Process a video file"""
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")

    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file_id}_input.mp4"
    output_path = OUTPUT_DIR / f"{file_id}_output.mp4"

    try:
        # Save uploaded video
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

        all_detections = []
        stats = {"frames_processed": 0, "total_damages": 0, "total_blurred": 0}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if mode == ProcessingMode.ROAD_DAMAGE:
                detections, frame = detect_road_damage(frame, confidence)
                stats["total_damages"] += len(detections)
                all_detections.extend(detections)

            elif mode == ProcessingMode.PRIVACY_BLUR:
                count, frame = detect_and_blur_privacy(
                    frame, blur_faces, blur_plates, blur_method.value, confidence
                )
                stats["total_blurred"] += count

            elif mode == ProcessingMode.COMBINED:
                count, frame = detect_and_blur_privacy(
                    frame, blur_faces, blur_plates, blur_method.value, 0.35
                )
                stats["total_blurred"] += count
                detections, frame = detect_road_damage(frame, confidence)
                stats["total_damages"] += len(detections)
                all_detections.extend(detections)

            writer.write(frame)
            stats["frames_processed"] += 1

        cap.release()
        writer.release()

        # Cleanup input
        input_path.unlink(missing_ok=True)

        return ProcessingResult(
            success=True,
            message=f"Processed {stats['frames_processed']} frames",
            detections=all_detections[:100] if all_detections else None,
            output_url=f"/outputs/{file_id}_output.mp4",
            stats=stats
        )

    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/outputs/{filename}")
async def get_output(filename: str):
    """Serve processed files"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")

    media_type = "video/mp4" if filename.endswith(".mp4") else "image/jpeg"
    return FileResponse(file_path, media_type=media_type)

@app.delete("/api/cleanup/{file_id}")
async def cleanup_files(file_id: str):
    """Clean up processed files"""
    for path in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in path.glob(f"{file_id}*"):
            f.unlink(missing_ok=True)
    return {"message": "Cleaned up"}

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Starting RoadPulse API...")
    load_models()
    print("RoadPulse API ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
