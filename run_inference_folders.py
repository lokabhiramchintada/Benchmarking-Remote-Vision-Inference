import os
import time
from pathlib import Path
import csv
from ultralytics import YOLO

MODEL_PATH = 'yolo12m.pt'
SOURCE_ROOT = 'images'
OUTPUT_ROOT = 'annotated'
CSV_FILE = 'inference_results.csv'

CONF = 0.25
IOU = 0.5

model = YOLO(MODEL_PATH)

SOURCE_ROOT = Path(SOURCE_ROOT)
OUTPUT_ROOT = Path(OUTPUT_ROOT)
OUTPUT_ROOT.mkdir(exist_ok=True)

rows = [["folder", "image_name", "num_detections", "inference_time_ms"]]

IMAGE_EXT = (".jpg", ".jpeg", ".png")

for subdir in sorted(SOURCE_ROOT.iterdir()):
    if not subdir.is_dir():
        continue

    images = [p for p in subdir.iterdir() if p.suffix.lower() in IMAGE_EXT]
    if not images:
        continue

    img = images[0]

    save_dir = OUTPUT_ROOT / subdir.name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / img.name

    print(f"🔍 Processing {subdir.name}/{img.name}")

    start = time.time()

    results = model(
        source=str(img),
        conf=CONF,
        iou=IOU,
        save=False,       # IMPORTANT
        verbose=False
    )

    end = time.time()

    detections = len(results[0].boxes)
    t_ms = (end - start) * 1000

    # 👇 THIS saves directly where we want
    results[0].save(filename=str(save_path))

    rows.append([subdir.name, img.name, detections, round(t_ms, 2)])

# save CSV
import csv
with open(CSV_FILE, "w", newline="") as f:
    csv.writer(f).writerows(rows)

print("\nDONE ✔")
