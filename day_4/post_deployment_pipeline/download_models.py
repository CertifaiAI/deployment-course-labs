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

import os
from pathlib import Path
import wget
import shutil


def download_model(source, target, filename):
    """ Download model from source to target directory """
    if not os.path.exists(target):
        os.mkdir(target) 
    target_file = str(Path(target).joinpath(filename))
    if os.path.exists(target_file):
        print('model already exists, skipping download')
        return
    print("Downloading from {} to {}".format(source, target))
    wget.download(source, target_file)  
    print("\nDone!")

def copy_model(source, target):
    if not os.path.exists(target):
        os.mkdir(target) 
    print("Copying model from {} to {}".format(source, target))
    shutil.copy(source, target)

if __name__ == "__main__":
    
    #### model_server folder ####
    source = 'https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/fruit_classifier_state_dict.pt'
    target = './model_server/app/artifacts'
    filename = 'fruit_classifier_state_dict.pt'
    download_model(source, target, filename)

    #### model_server_new folder ####
    source = 'https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/transfer_learning_model.pt'
    target = './model_server_new/app/artifacts'
    filename = 'transfer_learning_model.pt'
    download_model(source, target, filename)

    source = './model_server/app/artifacts/fruit_classifier_state_dict.pt'
    target = './model_server_new/app/artifacts/'
    copy_model(source, target)

    #### drift_detector_server folder ####
    source = 'https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/drift_detector.pkl'
    target = './drift_detector_server/app/artifacts'
    filename = 'drift_detector.pkl'
    download_model(source, target, filename)

    #### drift_detector_server_new folder ####
    source = './drift_detector_server/app/artifacts/drift_detector.pkl'
    target = './drift_detector_server_new/app/artifacts/'
    copy_model(source, target)