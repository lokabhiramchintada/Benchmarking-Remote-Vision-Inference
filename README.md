# Inference Latency Test

This project measures **end-to-end inference latency** for a YOLO object detection model running on a remote server. The client uploads images, receives detection results, and records:

* Upload latency
* Inference latency
* Download latency
* Total round-trip latency

Results are saved to a CSV file for comparison.

---

## 📂 Project Structure

```
.
├── client.py
├── server.py
├── input/
│   ├── test1.jpg
│   ├── test2.jpg
│   └── test3.jpg
└── yolo12m.pt
```

---

## 🖥️ Remote Access (Client → Server Tunnel)

Create an SSH tunnel so your local client can access the remote FastAPI server:

```bash
ssh -L 9000:localhost:8000 -J your_username@ada.iiit.ac.in your_username@gnode046
```

This forwards:

* Local port **9000**
* To remote **localhost:8000**

---

## 🚀 Start the Server

Run the FastAPI server on the remote machine:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 📦 Install Requirements

### Client

```bash
pip install requests pandas
```

### Server

```bash
pip install fastapi uvicorn ultralytics opencv-python numpy
```

---

## 🧠 YOLO Model

The server loads:

```
yolo12m.pt
```

Place this file in the same directory as `server.py`.

---

## 📡 Client Script (Latency Measurement)

The client uploads three images and logs latency metrics:

* Upload time
* Inference time (server-measured)
* Download time
* Total end-to-end time
* Number of detections

Results print as a table and save to:

```
latency_results.csv
```

---

## 📊 Output Example

```
===== FINAL COMPARISON =====

        Image     Description  Upload (ms)  Inference (ms)  Download (ms)  Total (ms)  Num Detections
   test1.jpg  Low detections         55.79           59.74          55.79      171.32              22
```

---

## 🧮 Latency Calculation Method

| Metric            | Source                  |
| ----------------- | ----------------------- |
| Total             | Client round-trip time  |
| Inference         | Server timestamps       |
| Upload + Download | Total − Inference       |
| Upload            | (Total − Inference) ÷ 2 |
| Download          | (Total − Inference) ÷ 2 |

This assumes symmetric network latency.

---

## 🛠️ API Endpoint

`POST /infer`

Returns JSON:

```json
{
  "server_receive_time": ...,
  "infer_start_time": ...,
  "infer_end_time": ...,
  "detections": [...]
}
```

---

## 📌 Notes

* Run server first
* Ensure SSH tunnel is active
* Client uses `http://localhost:9000/infer`
* Use Python 3.8+

---

## ✔️ Use Case

This setup is ideal for:

✅ Edge-cloud latency benchmarking
✅ Model deployment evaluation
✅ Network performance analysis
✅ Comparing workloads with different detection counts

---

