import random
import time

import requests

url = "http://127.0.0.1:5000/predict"

while True:
    # Generate random realistic sensor values
    sensor_data = {
        "Air temperature [K]": round(random.uniform(295, 310), 1),
        "Process temperature [K]": round(random.uniform(305, 320), 1),
        "Rotational speed [rpm]": round(random.uniform(1000, 3000), 0),
        "Torque [Nm]": round(random.uniform(20, 70), 1),
        "Tool wear [min]": round(random.uniform(0, 250), 0),
    }
    response = requests.post(url, json=sensor_data)
    print(f"Data: {sensor_data}")
    print(f"Prediction: {response.json()}\n")
    time.sleep(3)  # Wait 3 seconds between readings
