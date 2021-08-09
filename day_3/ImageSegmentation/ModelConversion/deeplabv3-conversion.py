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
Script to generate torchscript model for image segmentation app
prepared by: willadrsm
"""

import torch
import os

if __name__ == "__main__":
    model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    jit_script_model = torch.jit.script(model)

    target_path = "../App/app/src/main/assets/"
    filename = "model.pt"
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    torch.jit.save(jit_script_model, target_path + filename)
