#
#################################################################################
#
#  Copyright (c) 2021 CertifAI Sdn. Bhd.
#
#  This program is part of OSRFramework. You can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#################################################################################
#
#!/usr/bin/env python
# coding: utf-8 

from fastapi import FastAPI, Request, File, HTTPException
from fastapi.datastructures import UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import io
import logging
import sys
import argparse
import torch

# local import
from app.classifier import instantiate_net, instantiate_vgg16, inference
from app.models import ResponseDataModel
from app.utils import bytes_to_base64


app = FastAPI()
templates = Jinja2Templates("app/templates/")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

parser = argparse.ArgumentParser()
parser.add_argument("model", help="use particular model for deployment", type=str)
args = parser.parse_args()
if args.model.lower() == "vgg16":
    image_classifier = instantiate_vgg16()
    PORT = 8008
elif args.model.lower() == "custom":
    image_classifier = instantiate_net()
    PORT = 8009


@app.get("/")
def home():
    return (
        "Go to endpoint '/interface' or '/docs' or send HTTP POST request at '/predict'"
    )


@app.get("/interface/")
def interface(request: Request):
    return templates.TemplateResponse("interface.html", context={"request": request})


@app.post("/result/")
async def result(
    request: Request,
    file: UploadFile = File(...),
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    with torch.no_grad():
        inference_result = inference(image_classifier, image)
    logging.info(
        f"Predicted Class: {inference_result['predicted_class']} \t Probability: {inference_result['probability']}"
    )
    inference_image = "data:image/jpg;base64,{0}".format(bytes_to_base64(contents))
    return templates.TemplateResponse(
        "result.html",
        context={
            "request": request,
            "filename": file.filename,
            "content_type": file.content_type,
            "likely_class": inference_result["predicted_class"],
            "probability": inference_result["probability"].item(),
            "image_base64": inference_image,
        },
    )


@app.post("/predict", response_model=ResponseDataModel)
async def predict(file: UploadFile = File(...)):
    if file.content_type.startswith("image/") is False:
        raise HTTPException(
            status_code=400, detail=f"File '{file.filename}' is not an image."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        inference_result = inference(image_classifier, image)
        logging.info(
            f"Predicted Class: {inference_result['predicted_class']} \t Probability: {inference_result['probability']}"
        )
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "likely_class": inference_result["predicted_class"],
            "probability": inference_result["probability"],
            "image_base64": bytes_to_base64(contents),
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", port=PORT, host="0.0.0.0", reload=False)
