import cv2
import random
import pandas as pd
from ultralytics import YOLO
# -------------------------------
# Load YOLO Vehicle Model
# -------------------------------
model = YOLO("models/yolov8n.pt")

# -------------------------------
# Load VAHAN Database
# -------------------------------
vahan_db = pd.read_csv("database/vahan.csv")

# -------------------------------
# Load Image
# -------------------------------
image_path = "input/traffic5.jpg"
image = cv2.imread(image_path)

if image is None:
    print(" Image not found")
    exit()

# -------------------------------
# Run Vehicle Detection
# -------------------------------
results = model(image)

vehicle_classes = ["car", "motorcycle", "bus", "truck"]

for result in results:
    boxes = result.boxes
    names = result.names

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]

        if class_name not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # -------------------------------
        # 60% MATCH | 40% MISMATCH
        # -------------------------------
        chance = random.random()

        if chance < 0.6:
            # MATCH
            matched = vahan_db[vahan_db["type"] == class_name]
            if matched.empty:
                matched = vahan_db
            row = matched.sample(1).iloc[0]
            status = "MATCH"
            color = (0, 255, 0)
        else:
            # MISMATCH
            mismatched = vahan_db[vahan_db["type"] != class_name]
            if mismatched.empty:
                mismatched = vahan_db
            row = mismatched.sample(1).iloc[0]
            status = "MISMATCH"
            color = (0, 0, 255)

        plate_no = row["plate"]

        # -------------------------------
        # Draw Bounding Box
        # -------------------------------
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name.upper()} | {plate_no} | {status}"

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

# -------------------------------
# Save Output Image
# -------------------------------
output_path = "output/result.jpg"
cv2.imwrite(output_path, image)

print(" Vehicle detection completed")
print(" 60% MATCH and 40% MISMATCH simulated")
print(f" Output saved to {output_path}")
