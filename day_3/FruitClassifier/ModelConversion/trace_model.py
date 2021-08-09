#  Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
#  This program is part of OSRFramework. You can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version. You should have received a copy of the
#  GNU Affero General Public License along with this program.  If not, see
#  https://www.gnu.org/licenses/agpl-3.0
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Affero General Public License for more details.

"""
Script to generate torchscript model for fruit classifier app
prepared by: YCCertifai
"""

import os
from pathlib import Path
import wget
import torch
from model import Net
from torch.utils.mobile_optimizer import optimize_for_mobile


def download(source, target, filename):
    if not os.path.exists(target):
        os.mkdir(target)
    targt_file = str(Path(target).joinpath(filename))
    if os.path.exists(targt_file):
        print('data already exists, skipping download')
        return
    print("Downloading from {} to {}".format(source, target))
    wget.download(source, targt_file)
    print("\nDone!")


if __name__ == "__main__":
    # specify source
    source = 'https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/fruit_classifier_state_dict.pt'
    target = '../../../resources/'
    filename = 'fruit_classifier_state_dict.pt'
    download(source, target, filename)

    # Load model
    model = Net()
    model_path = target + filename
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Convert model to torchscript
    example = torch.rand(1, 3, 150, 150)
    traced_script_module = torch.jit.trace(model, example)
    optimized_traced_model = optimize_for_mobile(traced_script_module)

    target_folder = "../App/app/src/main/assets/"
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    optimized_traced_model.save("../App/app/src/main/assets/model.pt")
