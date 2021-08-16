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
from fastapi import FastAPI, HTTPException

import torch
import logging
import sys
import jsonpickle

# local import
from app.drift_detector import DriftDetector
from app.models import ResponseDataModel
from app.models import DriftDetectionRequest

from starlette_exporter import PrometheusMiddleware, handle_metrics
from app.monitoring import p_value_gauge

app = FastAPI()
drift_detector = DriftDetector()

"""Add instrumentation"""
app.add_middleware(
    PrometheusMiddleware,
    app_name = "drift_detector",
    prefix = "drift_detector",
    filter_unhandled_paths = True
    )
app.add_route("/metrics", handle_metrics)

@app.get("/")
def home():
    return "GET request"

@app.post("/driftdetect", response_model=ResponseDataModel)
async def driftdetect(drift_detection_req: DriftDetectionRequest):
    try:
        batch = jsonpickle.decode(drift_detection_req.batch_tensor)
        drift = False
        test_score, p_value = drift_detector.detect(torch.Tensor(batch))

        # update p-value gauge
        p_value_gauge.labels("current_p_value").set(p_value)

        if(p_value < drift_detection_req.p_value_threshold):
            drift = True

        return {
            "p_value": p_value,
            "drift": drift
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))
