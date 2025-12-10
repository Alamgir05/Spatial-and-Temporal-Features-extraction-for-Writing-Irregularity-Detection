# test_req.py
import requests

url = "http://127.0.0.1:8000/predict_json"   # or use your LAN IP: http://192.168.29.173:8000
payload = {
  "writer_id": "test",
  "sample_id": "s1",
  "strokes": [
    [
      {"x": 10, "y": 10, "t": 0.0, "pressure": 0.5, "penDown": True},
      {"x": 15, "y": 12, "t": 0.1, "pressure": 0.5, "penDown": True},
      {"x": 20, "y": 18, "t": 0.2, "pressure": 0.5, "penDown": True}
    ]
  ]
}

r = requests.post(url, json=payload, timeout=10)
print(r.status_code, r.text)
