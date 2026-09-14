import urllib.request
import json

try:
    req = urllib.request.Request(
        "http://localhost:8000/training/start",
        data=json.dumps({"epochs": 1, "batch_size": 64}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        print(f"Status Code: {response.status}")
        print(f"Response: {response.read().decode('utf-8')}")
except Exception as e:
    print(f"Error: {e}")
