import time
import requests
import pandas as pd

SERVER_URL = "http://localhost:9000/infer"

images = [
    ("input/test1.jpg", "Low detections"),
    ("input/test2.jpg", "High detections"),
    ("input/test3.jpg", "Zero detections"),
]

results = []

for img_path, desc in images:

    with open(img_path, "rb") as f:
        files = {"file": f}

        # ---- CLIENT TIMERS ----
        t_send = time.time()
        response = requests.post(SERVER_URL, files=files)
        t_receive = time.time()

    data = response.json()

    # ---- SERVER TIMERS ----
    t_infer_start = data["infer_start_time"]
    t_infer_end = data["infer_end_time"]

    # ---- SAFE LATENCIES ----
    total = (t_receive - t_send) * 1000
    infer = (t_infer_end - t_infer_start) * 1000

    # assume upload & download share remainder
    network = total - infer

    upload = network / 2
    download = network / 2

    results.append({
        "Image": img_path,
        "Description": desc,
        "Upload (ms)": round(upload, 2),
        "Inference (ms)": round(infer, 2),
        "Download (ms)": round(download, 2),
        "Total (ms)": round(total, 2),
        "Num Detections": len(data["detections"])
    })

df = pd.DataFrame(results)

print("\n===== FINAL COMPARISON =====\n")
print(df.to_string(index=False))

df.to_csv("latency_results.csv", index=False)
print("\nSaved latency_results.csv")
