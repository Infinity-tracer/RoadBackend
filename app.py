"""
RoadPulse API - FastAPI Backend for Road Damage Detection
Supports 3 modules: Road Damage Detection, Privacy Blur, Combined Processing
+ Accident Alert System for Bangalore routes
"""

import os
import uuid
import shutil
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
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
ACCIDENTS_DIR = Path("accidents")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ACCIDENTS_DIR.mkdir(exist_ok=True)

# Bangalore locations with coordinates (lat, lng)
BANGALORE_LOCATIONS = {
    "ArmyLayoutOldMadrasRoad": {"lat": 13.0180, "lng": 77.6500, "name": "Army Layout, Old Madras Road"},
    "BangloreChikkabalburHighway": {"lat": 13.1500, "lng": 77.5500, "name": "Bangalore-Chikkaballpur Highway"},
    "Banswadi": {"lat": 13.0150, "lng": 77.6400, "name": "Banaswadi"},
    "BengalurToAnantpurRoad": {"lat": 12.9200, "lng": 77.4500, "name": "Bengaluru-Anantapur Road"},
    "KRMarket": {"lat": 12.9622, "lng": 77.5788, "name": "KR Market"},
    "Whitefield": {"lat": 12.9698, "lng": 77.7500, "name": "Whitefield"},
    "ElectronicCity": {"lat": 12.8399, "lng": 77.6770, "name": "Electronic City"},
    "Koramangala": {"lat": 12.9352, "lng": 77.6245, "name": "Koramangala"},
    "Indiranagar": {"lat": 12.9784, "lng": 77.6408, "name": "Indiranagar"},
    "MGRoad": {"lat": 12.9756, "lng": 77.6062, "name": "MG Road"},
    "Jayanagar": {"lat": 12.9308, "lng": 77.5838, "name": "Jayanagar"},
    "Malleshwaram": {"lat": 13.0035, "lng": 77.5710, "name": "Malleshwaram"},
    "Hebbal": {"lat": 13.0358, "lng": 77.5970, "name": "Hebbal"},
    "Yelahanka": {"lat": 13.1007, "lng": 77.5963, "name": "Yelahanka"},
    "BTM": {"lat": 12.9166, "lng": 77.6101, "name": "BTM Layout"},
    "HSRLayout": {"lat": 12.9116, "lng": 77.6389, "name": "HSR Layout"},
    "Marathahalli": {"lat": 12.9591, "lng": 77.6974, "name": "Marathahalli"},
    "KRPuram": {"lat": 12.9988, "lng": 77.6960, "name": "KR Puram"},
    "Majestic": {"lat": 12.9772, "lng": 77.5713, "name": "Majestic"},
    "Yeshwanthpur": {"lat": 13.0280, "lng": 77.5500, "name": "Yeshwanthpur"},
}

# In-memory accident storage (in production, use a database)
active_accidents: Dict[str, dict] = {}

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

class AccidentReport(BaseModel):
    id: str
    location: str
    location_name: str
    lat: float
    lng: float
    timestamp: str
    severity: str
    video_url: Optional[str] = None
    active: bool = True

class RouteCheckRequest(BaseModel):
    start_location: str
    end_location: str

class RouteCheckResponse(BaseModel):
    has_accidents: bool
    accidents: List[AccidentReport]
    route_info: dict

# Global model storage
models = {}

def get_model(model_name: str):
    """Lazy load models only when needed"""
    global models

    if model_name in models:
        return models[model_name]

    model_paths = {
        "road_damage": ROAD_DAMAGE_MODEL_PATH,
        "face": FACE_MODEL_PATH,
        "plate": PLATE_MODEL_PATH
    }

    path = model_paths.get(model_name)
    if path and path.exists():
        print(f"Loading {model_name} model...")
        models[model_name] = YOLO(str(path))
        print(f"Loaded {model_name} model from {path}")
        return models[model_name]

    return None

def load_models():
    """Check if models exist (don't load them yet to save memory)"""
    print("Models will be loaded on-demand to save memory")
    if ROAD_DAMAGE_MODEL_PATH.exists():
        print(f"Road damage model available at {ROAD_DAMAGE_MODEL_PATH}")
    if FACE_MODEL_PATH.exists():
        print(f"Face detection model available at {FACE_MODEL_PATH}")
    if PLATE_MODEL_PATH.exists():
        print(f"License plate model available at {PLATE_MODEL_PATH}")

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
    model = get_model("road_damage")
    if model is None:
        return [], frame

    h_ori, w_ori = frame.shape[:2]
    frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

    results = model.predict(frame_resized, conf=conf_threshold, verbose=False)

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

    if blur_faces:
        face_model = get_model("face")
        if face_model:
            results = face_model(frame, conf=conf_threshold, verbose=False)
            if results and results[0].boxes is not None:
                for box in results[0].boxes.xyxy.cpu().numpy():
                    all_boxes.append(tuple(map(int, box)))

    if blur_plates:
        plate_model = get_model("plate")
        if plate_model:
            results = plate_model(frame, conf=conf_threshold, verbose=False)
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

# ============ ACCIDENT ALERT SYSTEM ENDPOINTS ============

def extract_location_from_filename(filename: str) -> Optional[dict]:
    """Extract location from video filename"""
    name = Path(filename).stem
    for loc_key, loc_data in BANGALORE_LOCATIONS.items():
        if loc_key.lower() in name.lower() or name.lower() in loc_key.lower():
            return {"key": loc_key, **loc_data}
    return None

def detect_accident_in_video(video_path: str, conf_threshold: float = 0.5) -> dict:
    """Simple accident detection using motion/anomaly detection"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"detected": False, "severity": "unknown", "confidence": 0}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    prev_frame = None
    motion_scores = []

    sample_interval = max(1, frame_count // 30)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)

            prev_frame = gray
        frame_idx += 1

    cap.release()

    if not motion_scores:
        return {"detected": False, "severity": "unknown", "confidence": 0.0, "motion_score": 0.0}

    avg_motion = float(np.mean(motion_scores))
    max_motion = float(np.max(motion_scores))

    # Determine if accident detected based on sudden motion changes
    detected = bool(max_motion > 30 or avg_motion > 15)

    if max_motion > 50:
        severity = "severe"
        confidence = 0.9
    elif max_motion > 35:
        severity = "moderate"
        confidence = 0.75
    elif max_motion > 20:
        severity = "minor"
        confidence = 0.6
    else:
        severity = "unknown"
        confidence = 0.4

    return {
        "detected": detected,
        "severity": severity,
        "confidence": float(confidence),
        "motion_score": float(max_motion)
    }

@app.post("/api/accident/upload")
async def upload_accident_video(
    file: UploadFile = File(...),
    location: Optional[str] = Form(default=None)
):
    """Upload dashcam video and detect accident"""
    # Ensure accidents directory exists
    ACCIDENTS_DIR.mkdir(exist_ok=True)

    # Check file extension if content_type is not reliable
    filename = file.filename or "video.mp4"
    is_video = (
        (file.content_type and file.content_type.startswith("video/")) or
        filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    )

    if not is_video:
        raise HTTPException(400, "File must be a video")

    accident_id = str(uuid.uuid4())[:8]
    safe_filename = file.filename.replace(" ", "_") if file.filename else "video.mp4"
    input_path = ACCIDENTS_DIR / f"{accident_id}_{safe_filename}"

    try:
        # Save uploaded video
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # Extract location from filename if not provided
        loc_info = None
        if location and location in BANGALORE_LOCATIONS:
            loc_info = {"key": location, **BANGALORE_LOCATIONS[location]}
        else:
            loc_info = extract_location_from_filename(safe_filename)

        if not loc_info:
            # Default to a central Bangalore location
            loc_info = {
                "key": "Unknown",
                "lat": 12.9716,
                "lng": 77.5946,
                "name": "Bangalore (Unknown Location)"
            }

        # Detect accident
        detection_result = detect_accident_in_video(str(input_path))

        # Create accident report
        accident_report = AccidentReport(
            id=accident_id,
            location=loc_info["key"],
            location_name=loc_info["name"],
            lat=loc_info["lat"],
            lng=loc_info["lng"],
            timestamp=datetime.now().isoformat(),
            severity=detection_result["severity"],
            video_url=f"/accidents/{input_path.name}",
            active=detection_result["detected"]
        )

        # Store in active accidents
        if detection_result["detected"]:
            active_accidents[accident_id] = accident_report.model_dump()

        return {
            "success": True,
            "accident_detected": detection_result["detected"],
            "accident": accident_report.model_dump(),
            "detection_info": detection_result,
            "message": f"Accident {'detected' if detection_result['detected'] else 'not detected'} at {loc_info['name']}"
        }

    except Exception as e:
        import traceback
        print(f"Error in upload_accident_video: {traceback.format_exc()}")
        raise HTTPException(500, str(e))

@app.get("/api/accident/active")
async def get_active_accidents():
    """Get all active accidents"""
    return {
        "count": len(active_accidents),
        "accidents": list(active_accidents.values())
    }

@app.post("/api/route/check")
async def check_route_for_accidents(request: RouteCheckRequest):
    """Check if route has any active accidents"""
    start = request.start_location
    end = request.end_location

    start_info = BANGALORE_LOCATIONS.get(start)
    end_info = BANGALORE_LOCATIONS.get(end)

    if not start_info:
        # Try to find partial match
        for key, val in BANGALORE_LOCATIONS.items():
            if start.lower() in key.lower() or start.lower() in val["name"].lower():
                start_info = val
                start = key
                break

    if not end_info:
        for key, val in BANGALORE_LOCATIONS.items():
            if end.lower() in key.lower() or end.lower() in val["name"].lower():
                end_info = val
                end = key
                break

    if not start_info or not end_info:
        return RouteCheckResponse(
            has_accidents=False,
            accidents=[],
            route_info={"error": "Could not find one or both locations"}
        )

    # Find accidents on or near the route
    accidents_on_route = []

    for acc_id, accident in active_accidents.items():
        acc_lat = accident["lat"]
        acc_lng = accident["lng"]

        # Simple check: is accident location between start and end?
        lat_range = sorted([start_info["lat"], end_info["lat"]])
        lng_range = sorted([start_info["lng"], end_info["lng"]])

        # Add buffer zone of ~2km
        buffer = 0.02

        if (lat_range[0] - buffer <= acc_lat <= lat_range[1] + buffer and
            lng_range[0] - buffer <= acc_lng <= lng_range[1] + buffer):
            accidents_on_route.append(AccidentReport(**accident))

    return RouteCheckResponse(
        has_accidents=len(accidents_on_route) > 0,
        accidents=accidents_on_route,
        route_info={
            "start": {"name": start_info["name"], "lat": start_info["lat"], "lng": start_info["lng"]},
            "end": {"name": end_info["name"], "lat": end_info["lat"], "lng": end_info["lng"]},
            "accidents_found": len(accidents_on_route)
        }
    )

@app.get("/api/locations")
async def get_bangalore_locations():
    """Get all Bangalore locations for dropdown"""
    return {
        "locations": [
            {"key": key, "name": val["name"], "lat": val["lat"], "lng": val["lng"]}
            for key, val in BANGALORE_LOCATIONS.items()
        ]
    }

@app.delete("/api/accident/{accident_id}")
async def clear_accident(accident_id: str):
    """Clear/resolve an accident"""
    if accident_id in active_accidents:
        del active_accidents[accident_id]
        return {"success": True, "message": "Accident cleared"}
    raise HTTPException(404, "Accident not found")

# Mount static files
app.mount("/accidents", StaticFiles(directory="accidents"), name="accidents")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Starting RoadPulse API...")
    # Ensure all directories exist
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    ACCIDENTS_DIR.mkdir(exist_ok=True)
    load_models()
    print("RoadPulse API ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
