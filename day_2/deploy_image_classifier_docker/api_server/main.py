from fastapi import FastAPI, HTTPException
import uvicorn
import sys
import logging
import os

# local import
from api_server.utils import base64_str_to_PILImage

from api_server.classifier import SqueezeNet
from api_server.models import Base64str


app = FastAPI()
image_classifer = SqueezeNet()


@app.get("/")
def home():
    return "Hello!"


@app.post("/predict")
def predict(payload: Base64str):
    image = base64_str_to_PILImage(payload.base64str)
    try:
        predicted_class = image_classifer.predict(image)
        logging.info(f"Predicted Class: {predicted_class}")

        return {"likely_class": predicted_class}

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api_server.main:app", reload=False, host="0.0.0.0", port=8000, log_level="info"
    )
