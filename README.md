# RoadPulse API Backend

FastAPI backend for the RoadPulse road damage detection and privacy protection system.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models
python download_models.py

# Run server
uvicorn app:app --reload --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Deployment to Render

### Option 1: Blueprint (Recommended)
```bash
# From the backend directory
render blueprint create
```

### Option 2: Manual
1. Create new Web Service on Render
2. Connect GitHub repository
3. Set environment:
   - **Build**: `pip install -r requirements.txt && python download_models.py`
   - **Start**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Docker

```bash
# Build
docker build -t roadpulse-api .

# Run
docker run -p 8000:8000 roadpulse-api
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8000 |

## Models

The following models are downloaded automatically:
- `YOLOv8_Small_RDD.pt` - Road damage detection (89MB)
- `yolov8n-face.pt` - Face detection (6MB)
- `license_plate_detector.pt` - Plate detection (optional)
