
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
import requests
import json
import datetime
import jsonpickle
import torch

P_VALUE_THRESHOLD = 0.05
BATCH_SIZE = 5
batch_tensor = torch.rand(BATCH_SIZE, 18496)

drift_detection_req = json.dumps({
                    "p_value_threshold": P_VALUE_THRESHOLD,
                    "batch_tensor": jsonpickle.encode(batch_tensor.detach().numpy())
                })
response = requests.post("http://localhost:8009/driftdetect", data=drift_detection_req)
result = response.json()
print("Response headers: ", response.headers)
print("Result: ", result)