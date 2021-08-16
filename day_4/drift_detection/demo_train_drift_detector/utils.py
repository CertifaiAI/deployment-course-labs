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
import torch
import zipfile

def download_dataset(source, target, filename):
    if not os.path.exists(target):
        os.mkdir(target)
    target_file = str(Path(target).joinpath(filename))
    if os.path.exists(target_file):
        print('data already exists, skipping download')
        return
    print("Downloading from {} to {}".format(source, target))
    wget.download(source, target_file)
    print("\nDone!")
    print('Unzipping {}'.format(target_file))
    zipr = zipfile.ZipFile(target_file)
    zipr.extractall(target)
    zipr.close()
    print('Done!')

def download_model(source, target, filename):
    """ Download model from source to target directory """
    if not os.path.exists(target):
        os.mkdir(target)
    targt_file = str(Path(target).joinpath(filename))
    if os.path.exists(targt_file):
        print('model already exists, skipping download')
        return
    print("Downloading from {} to {}".format(source, target))
    wget.download(source, targt_file)
    print("\nDone!")

def load_model_state_dict(model, model_state_dict_path):
    """
    A function to load model from specified state dict path location.

    Parameters:
    model_path (String): state dict path location

    Returns:
    Loaded model
    """
    state_dict = torch.load(model_state_dict_path)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model