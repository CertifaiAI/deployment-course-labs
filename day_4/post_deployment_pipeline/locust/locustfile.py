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

from locust import HttpUser, task, between
from pathlib import Path
import random
import glob

all_image_path = []
pattern = Path(__file__).resolve().parents[3]/"resources/data/fruits_image_classification/dirty_test/*/**.jpg"
for path in glob.glob(str(pattern)):
    all_image_path.append(Path(path))

all_image_path2 = []
pattern2 = Path(__file__).resolve().parents[3]/"resources/data/fruits_image_classification/test/*/**.jpg"
for path2 in glob.glob(str(pattern2)):
    all_image_path2.append(Path(path2))


class PredictionUser(HttpUser):
    # Wait time between requests
    wait_time = between(0.5,3)

    @task(1)
    def healthcheck(self):
        self.client.get('/')
    
    # 200 - Legitimate request - From Dirty Test Folder
    @task(10)
    def prediction(self):
        imagePath = random.choice(all_image_path)
        files_dict = {
            'file':(imagePath.name,open(imagePath,'rb'),'image/jpeg')
            }
        self.client.post('/predict',files=files_dict)
    
    # 200 - Legitimate request - From Test Folder
    @task(10)
    def prediction(self):
        imagePath = random.choice(all_image_path2)
        files_dict = {
            'file':(imagePath.name,open(imagePath,'rb'),'image/jpeg')
            }
        self.client.post('/predict',files=files_dict)
    
    # 4xx - Bad requests: Specify a media type to purposefully raise 4xx error
    @task(1)
    def prediction_bad_4xx(self):
        imagePath = random.choice(all_image_path)
        files_dict = {
            'file':(imagePath.name,open(imagePath,'rb'),'fake_client_error/jpeg')
            }
        self.client.post('/predict',files=files_dict)
    
    # 5xx - Bad requests: Specify a media type to purposefully raise 5xx error
    @task(1)
    def prediction_bad_5xx(self):
        imagePath = random.choice(all_image_path)
        files_dict = {
            'file':(imagePath.name,open(imagePath,'rb'),'fake_server_error/jpeg')
            }
        self.client.post('/predict',files=files_dict)