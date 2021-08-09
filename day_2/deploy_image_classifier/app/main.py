from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from PIL import Image
import io
import logging
import sys

# local import
from app.classifier import ImageClassifier
from app.models import ResponseDataModel

app = FastAPI()
image_classifier = ImageClassifier()


@app.get("/")
def home():
    return "Hello!"


@app.post("/predict", response_model=ResponseDataModel)
async def predict(file: UploadFile = File(...)):
    if file.content_type.startswith("image/") is False:
        raise HTTPException(
            status_code=400, detail=f"File '{file.filename}' is not an image."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        predicted_class = image_classifier.predict(image)

        logging.info(f"Predicted Class: {predicted_class}")
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "likely_class": predicted_class,
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, log_level="info")
