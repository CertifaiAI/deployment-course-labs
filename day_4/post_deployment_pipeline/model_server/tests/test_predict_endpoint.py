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
from pathlib import Path

FILENAME = "image3.jpg"
FILE_DIRECTORY = "apple/"
DATA_DIRECTORY = Path(__file__).resolve().parents[4] /"resources/data/fruits_image_classification/test"
IMAGE_PATH = DATA_DIRECTORY/FILE_DIRECTORY/FILENAME

TEST_IMAGE = {
    'file':(IMAGE_PATH.name, open(IMAGE_PATH,'rb'),'image/jpeg')
}

response = requests.post("http://localhost:8000/predict", files=TEST_IMAGE)
result = response.json()
print("Response headers: ", response.headers)
print("Inference file: ", FILENAME)
print("Result: ", result)