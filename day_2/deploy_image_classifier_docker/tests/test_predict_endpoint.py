import requests
import json
import base64
import os


FILENAME = "Chest.jpeg"
FILE_DIRECTORY = "COVID-19/"
DATA_DIRECTORY = os.environ["PYTHONPATH"] + "/sample_data/"

with open(DATA_DIRECTORY+FILE_DIRECTORY+FILENAME, "rb") as image:
    base64str = base64.b64encode(image.read()).decode("utf-8")

payload = json.dumps({
    "base64str": base64str
})
response = requests.post("http://localhost:8000/predict", data=payload)
result = response.json()
print("Response headers: ", response.headers)
print("Inference file: ", FILENAME)
print("Result: ", result)