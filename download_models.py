"""
Download required YOLO models for RoadPulse API
"""
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "YOLOv8_Small_RDD.pt": {
        "url": "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt",
        "size": 89569358
    },
    "yolov8n-face.pt": {
        "url": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
        "size": None
    }
}

def download_file(url: str, dest: Path, expected_size: int = None):
    """Download a file with progress"""
    if dest.exists():
        if expected_size and dest.stat().st_size == expected_size:
            print(f"[OK] {dest.name} already exists")
            return

    print(f"[DOWNLOADING] {dest.name} from {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] Downloaded {dest.name}")
    except Exception as e:
        print(f"[ERROR] Failed to download {dest.name}: {e}")

def main():
    print("=" * 50)
    print("RoadPulse Model Downloader")
    print("=" * 50)

    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        download_file(info["url"], dest, info.get("size"))

    print("=" * 50)
    print("Model download complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
