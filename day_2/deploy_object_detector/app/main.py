from fastapi import FastAPI, Request, Form, File
from fastapi.datastructures import UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch

# local import
from app.models import Base64str
from app.detector import (
    model,
    extract_results,
    draw_bounding_boxes,
)
from app.utils import preprocessing, base64str_to_PILImage, bytes_to_base64


app = FastAPI()
object_detector = model
templates = Jinja2Templates("app/templates/")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def home():
    return "Go to endpoint '/interface' or '/docs' or send HTTP request at '/predict'"


@app.get("/interface/")
def interface(request: Request):
    return templates.TemplateResponse("interface.html", context={"request": request})


@app.post("/result/")
async def result(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(...),
):
    image_bytes = await file.read()
    image_base64 = bytes_to_base64(image_bytes)
    image = base64str_to_PILImage(image_base64)
    image_tensor = preprocessing(image)

    with torch.no_grad():
        raw_inferences = object_detector([image_tensor])
    processed_inferences = extract_results(raw_inferences, threshold)
    data_uri = draw_bounding_boxes(
        image, processed_inferences["boxes"], processed_inferences["classes"]
    )
    inference_output = "data:image/jpg;base64,{0}".format(data_uri)

    return templates.TemplateResponse(
        "result.html",
        context={
            "request": request,
            "result": processed_inferences["classes"],
            "image": inference_output,
        },
    )


@app.post("/predict")
def predict(input_data: Base64str):
    """
    FastAPI API will take a base 64 image as input and return a json object
    """

    image = base64str_to_PILImage(input_data.base64str)
    image_tensor = preprocessing(image)

    # get prediction on image and setting gradient computation to false
    with torch.no_grad():
        raw_inferences = object_detector([image_tensor])
    processed_inferences = extract_results(raw_inferences, input_data.threshold)
    draw_bounding_boxes(
        image, processed_inferences["boxes"], processed_inferences["classes"]
    )

    return processed_inferences


if __name__ == "__main__":
    uvicorn.run("main:app", reload=False, host="0.0.0.0", port=8000)
