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

import os
from pathlib import Path
import wget


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


if __name__ == "__main__":
    link = [
        'https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/fruit_classifier_state_dict.pt', 
        'https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/transfer_learning_model.pt'
    ]
    targets = ['./app_custom/app/artifacts', './app_vgg16/app/artifacts']
    filenames = ['fruit_classifier_state_dict.pt', 'transfer_learning_model.pt']
    
    # downloading
    for source, target, filename in zip(link, targets, filenames):
        download_model(source, target, filename)