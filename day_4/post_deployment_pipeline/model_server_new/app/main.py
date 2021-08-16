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

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import logging
import sys
import pickle
import datetime
from datetime import timezone
from pymongo import MongoClient
from bson.binary import Binary

from app.torch_models import ImageClassifier, FeatureExtractor
from app.models import ResponseDataModel
from app.utils import mongo_insert_doc

from starlette_exporter import PrometheusMiddleware, handle_metrics
from app.monitoring import predicted_class_counter

app = FastAPI()
image_classifier = ImageClassifier()
feature_extractor = FeatureExtractor()

# Uncomment line with host='localhost' when testing
mongodbClient = MongoClient(host='mongodb',port=27017,username='admin', password='admin', authSource='drift_detection')
# mongodbClient = MongoClient(host='localhost',port=27017,username='admin', password='admin', authSource='drift_detection')
mongodb = mongodbClient["drift_detection"]

"""Add instrumentation"""
app.add_middleware(
    PrometheusMiddleware,
    app_name = "fastapi",
    prefix = "fastapi",
    filter_unhandled_paths = True
    )
app.add_route("/metrics", handle_metrics)

@app.get("/")
def home():
    return "GET request"


@app.post("/predict", response_model=ResponseDataModel)
async def predict(file: UploadFile = File(...)):
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Simulate server error 5xx
        if file.content_type.startswith("fake_server_error/") is True:
            raise HTTPException(
                status_code=500, detail="Successful simulation of server error 5xx."
            )

        if file.content_type.startswith("image/") is False:
            raise HTTPException(
                status_code=400, detail=f"File '{file.filename}' is not an image."
            )

        try:
            # predict class of image
            predicted_class = image_classifier.predict(image)

            # increment prediction class counter
            predicted_class_counter.labels(predicted_class).inc()

            # extract image features
            image_features = await feature_extractor.compute_features(image)

            # insert extracted image features into mongodb            
            await mongo_insert_doc(mongodb, {"image_features_new": Binary(pickle.dumps(image_features, protocol=2)), "created_at": datetime.datetime.now(timezone.utc) })

            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "likely_class": predicted_class,
            }

        except Exception as error:
            logging.exception(error)
            e = sys.exc_info()[1]
            raise HTTPException(status_code=500, detail=str(e))