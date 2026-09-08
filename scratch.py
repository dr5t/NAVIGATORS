import json

with open("data/raw_trips/trip_4.json", "r") as f:
    data = json.load(f)
    print("Keys in trip_4.json:", data.keys())
    if "data" in data and "gnss" in data["data"]:
        print("GNSS data available. Length:", len(data["data"]["gnss"]))
        print("First coordinate:", data["data"]["gnss"][0])
    else:
        print("GNSS key not found in data['data']")
