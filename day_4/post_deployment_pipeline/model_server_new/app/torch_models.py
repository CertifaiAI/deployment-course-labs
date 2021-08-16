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

from app.config import NUM_CLASSES, CLASSES, TRANSFORM, MODEL_WEIGHT_PATH, NEW_MODEL_WEIGHT_PATH
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

def instantiate_vgg16():
    """Instantiate an VGG16 object"""
    vgg16 = models.vgg16(pretrained=False)
    vgg16.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES),
        nn.Softmax(dim=1),
    )
    # load_model_weights(vgg16, os.path.join(MODEL_B))
    return vgg16
    
def load_model_weights(model, model_weights):
    """Load model weights from serialized weights by reference"""
    model.load_state_dict(torch.load(model_weights, map_location=torch.device("cpu")))

class ImageClassifier:
    """Blueprint/Construct for image classifier"""

    def __init__(self):
        self.classifier = instantiate_vgg16()
        self.classifier.load_state_dict(
            torch.load(NEW_MODEL_WEIGHT_PATH, map_location=torch.device("cpu"))
        )
        self.classifier.eval()

    def predict(self, image):
        image = TRANSFORM(image)
        image = image.unsqueeze(0)
        output = self.classifier(image)
        class_index = torch.argmax(output, dim=1)

        return CLASSES[class_index]

class Net(nn.Module):
    """Model architecture that is used in Day 1"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_CLASSES, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(18496, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FeatureExtractor:
    """Extends Net architecture but remove FC layers"""

    def __init__(self):
        self.feature_extractor = Net()
        self.feature_extractor.load_state_dict(
            torch.load(MODEL_WEIGHT_PATH, map_location=torch.device("cpu"))
        )
        self.feature_extractor.fc1 = torch.nn.Identity()
        self.feature_extractor.fc2 = torch.nn.Identity()
        self.feature_extractor.fc3 = torch.nn.Identity()
        self.feature_extractor.eval()
        
    async def compute_features(self, image):
        
        image = TRANSFORM(image)
        image = image.unsqueeze(0)

        # extract features using the same architecture as image classifier 
        features = self.feature_extractor(image)
        return features