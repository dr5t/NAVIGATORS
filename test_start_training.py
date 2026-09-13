import requests

try:
    response = requests.post("http://localhost:8000/training/start", json={"epochs": 1, "batch_size": 64})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
