"""
FastAPI backend for AI-based Vehicle Verification System.

Features:
- Accepts image upload from frontend.
- Runs YOLOv8 vehicle detection (car, motorcycle, bus, truck).
- Simulates VAHAN verification against a CSV "database".
- Draws green (MATCH) / red (MISMATCH) bounding boxes.
- Saves processed image in the output folder.
- Returns JSON with image URL and match statistics.
"""

import random
import uuid
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


# -----------------------------
# Paths and folder setup
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # Points to Vehicle_Prototype/
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# YOLO model path (you should place yolov8n.pt in the models/ folder)
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

# VAHAN CSV database path (you should place vahan.csv in the database/ folder)
VAHAN_CSV_PATH = DATABASE_DIR / "vahan.csv"


# -----------------------------
# FastAPI app and CORS
# -----------------------------

app = FastAPI(title="AI Vehicle Verification API")

# Enable CORS so the frontend (running on a different origin) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve processed images from /output URL
app.mount(
    "/output",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="output",
)


# -----------------------------
# Model and database loading
# -----------------------------

def load_yolo_model() -> YOLO:
    """
    Load YOLOv8 model from the models directory.
    """
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found at {YOLO_MODEL_PATH}. "
            f"Please download yolov8n.pt and place it in the models/ folder."
        )
    return YOLO(str(YOLO_MODEL_PATH))


def load_vahan_database() -> pd.DataFrame:
    """
    Load the VAHAN CSV database.

    For this prototype we only use it to simulate verification,
    so any basic CSV with a 'registration_number' column is fine.
    """
    if not VAHAN_CSV_PATH.exists():
        # If the CSV is missing, return an empty DataFrame and still work.
        return pd.DataFrame()
    return pd.read_csv(VAHAN_CSV_PATH)


model = load_yolo_model()
vahan_df = load_vahan_database()


# -----------------------------
# Detection helpers
# -----------------------------

# COCO class IDs for vehicles in YOLOv8
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def run_vehicle_detection(image_path: Path) -> List[Dict[str, Any]]:
    """
    Run YOLOv8 on the image and return a list of vehicle detections.

    Each detection is a dict with:
      - class_id
      - class_name
      - confidence
      - bbox (x1, y1, x2, y2)
    """
    results = model(str(image_path))

    detections: List[Dict[str, Any]] = []

    if not results:
        return detections

    result = results[0]
    boxes = result.boxes

    if boxes is None:
        return detections

    # Convert tensors to numpy
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    for i in range(len(xyxy)):
        class_id = int(cls[i])
        if class_id not in COCO_VEHICLE_CLASSES:
            # Skip non-vehicle classes
            continue

        x1, y1, x2, y2 = xyxy[i]
        detections.append(
            {
                "class_id": class_id,
                "class_name": COCO_VEHICLE_CLASSES[class_id],
                "confidence": float(conf[i]),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            }
        )

    return detections


def simulate_vahan_verification(
    detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Simulate VAHAN verification using the CSV "database".

    For this prototype:
    - We assign 60% of vehicles as MATCH and 40% as MISMATCH randomly.
    - We don't use real plate recognition; this is just a demo.

    The function returns an updated list of detection dicts with:
      - status: "MATCH" or "MISMATCH"
    """
    verified: List[Dict[str, Any]] = []

    for det in detections:
        is_match = random.random() < 0.60  # 60% probability of MATCH
        status = "MATCH" if is_match else "MISMATCH"

        det_with_status = {**det, "status": status}
        verified.append(det_with_status)

    return verified


def draw_bounding_boxes(
    image: np.ndarray, detections: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Draw green (MATCH) and red (MISMATCH) bounding boxes on the image.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class_name"]
        status = det.get("status", "UNKNOWN")
        conf = det.get("confidence", 0.0)

        # Green for MATCH, Red for MISMATCH
        if status == "MATCH":
            color = (0, 255, 0)  # BGR (green)
        elif status == "MISMATCH":
            color = (0, 0, 255)  # BGR (red)
        else:
            color = (255, 255, 0)  # BGR (cyan) for unknown

        label = f"{status} {cls_name} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        # Draw label background
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - label_height - baseline - 4),
            (x1 + label_width + 4, y1),
            color,
            thickness=-1,
        )

        # Put text
        cv2.putText(
            image,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return image


# -----------------------------
# API endpoint
# -----------------------------

@app.post("/detect")
async def detect_vehicles(file: UploadFile = File(...)) -> JSONResponse:
    """
    Detect vehicles in an uploaded image and simulate VAHAN verification.

    Steps:
    1. Save uploaded image into input/ folder.
    2. Run YOLOv8 vehicle detection.
    3. Simulate MATCH / MISMATCH decision (60% / 40%).
    4. Draw bounding boxes and save result in output/ folder.
    5. Return JSON with processed image URL and summary counts.
    """
    # Basic file type check
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Create unique filename for the uploaded image
    file_ext = Path(file.filename).suffix or ".jpg"
    input_filename = f"input_{uuid.uuid4().hex}{file_ext}"
    input_path = INPUT_DIR / input_filename

    # Save uploaded image to disk
    contents = await file.read()
    with input_path.open("wb") as f:
        f.write(contents)

    # Read image with OpenCV (BGR format)
    image = cv2.imread(str(input_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    # 1) YOLOv8 detection
    detections = run_vehicle_detection(input_path)

    # 2) Simulate VAHAN verification (MATCH/MISMATCH)
    verified_detections = simulate_vahan_verification(detections)

    # 3) Draw bounding boxes on a copy of the image
    image_with_boxes = draw_bounding_boxes(image.copy(), verified_detections)

    # 4) Save processed image into output/ folder
    output_filename = f"result_{uuid.uuid4().hex}.jpg"
    output_path = OUTPUT_DIR / output_filename

    # Use JPEG format for output
    cv2.imwrite(str(output_path), image_with_boxes)

    # 5) Build summary statistics
    total = len(verified_detections)
    match_count = sum(1 for d in verified_detections if d["status"] == "MATCH")
    mismatch_count = sum(1 for d in verified_detections if d["status"] == "MISMATCH")

    # This is the URL path that the frontend can use to load the processed image.
    image_url = f"/output/{output_filename}"

    # Minimal JSON (matches your example)
    response_payload: Dict[str, Any] = {
      "image_url": image_url,
      "total": total,
      "match": match_count,
      "mismatch": mismatch_count,
    }

    # Extra fields for a richer frontend (optional)
    response_payload["summary"] = {
        "total_vehicles": total,
        "match_count": match_count,
        "mismatch_count": mismatch_count,
        "vehicles": [
            {
                "id": idx + 1,
                "type": det["class_name"],
                "status": det["status"],
                "confidence": det["confidence"],
                "bbox": {
                    "x1": det["bbox"][0],
                    "y1": det["bbox"][1],
                    "x2": det["bbox"][2],
                    "y2": det["bbox"][3],
                },
            }
            for idx, det in enumerate(verified_detections)
        ],
    }

    # Backwards-compatible key for earlier frontend expectation
    response_payload["processed_image_url"] = image_url

    return JSONResponse(response_payload)


# -----------------------------
# Development entry point
# -----------------------------

if __name__ == "__main__":
    # Command to run the server for development:
    #   python backend/app.py
    #
    # Or directly with uvicorn from the project root (Vehicle_Prototype/):
    #   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

