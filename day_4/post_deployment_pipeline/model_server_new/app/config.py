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
 
import torchvision.transforms as transforms
import os

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)
NUM_CLASSES = 3
CLASSES = ["apple", "grapes", "lemon"]
FILENAME = "fruit_classifier_state_dict.pt"
NEW_MODEL_FILENAME = "transfer_learning_model.pt"
FILE_PATH = os.path.realpath(__file__)
MODEL_WEIGHT_PATH = os.path.join(os.path.dirname(FILE_PATH), "artifacts", FILENAME)
NEW_MODEL_WEIGHT_PATH = os.path.join(os.path.dirname(FILE_PATH), "artifacts", NEW_MODEL_FILENAME)