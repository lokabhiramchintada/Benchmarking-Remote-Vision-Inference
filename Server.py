import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
import numpy as np
import cv2

app = FastAPI()

print("Loading model…")
model = YOLO("yolo12m.pt")
print("Model loaded.")

@app.post("/infer")
async def infer(file: UploadFile = File(...)):

    # -------- server receive timestamp --------
    t_server_recv = time.time()

    img_bytes = await file.read()
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # -------- inference timestamps --------
    t_infer_start = time.time()
    results = model(img)
    t_infer_end = time.time()

    detections = []
    for b in results[0].boxes:
        detections.append({
            "x1": float(b.xyxy[0][0]),
            "y1": float(b.xyxy[0][1]),
            "x2": float(b.xyxy[0][2]),
            "y2": float(b.xyxy[0][3]),
            "class_id": int(b.cls[0]),
            "confidence": float(b.conf[0])
        })

    return JSONResponse({
        "server_receive_time": t_server_recv,
        "infer_start_time": t_infer_start,
        "infer_end_time": t_infer_end,
        "detections": detections
    })


if __name__ == "__main__":
    uvicorn.run("server:app",
                host="0.0.0.0",
                port=9000,
                workers=1)
