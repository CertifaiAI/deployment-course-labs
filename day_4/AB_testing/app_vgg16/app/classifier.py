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

from app.config import *
import torch
from torch import nn
import torchvision.models as models
import os


class Net(nn.Module):
    """Model architecture that is used in Day 1"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(34 * 34 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))


def load_model_weights(model, model_weights):
    """Load model weights from serialized weights by reference"""
    model.load_state_dict(torch.load(model_weights, map_location=torch.device("cpu")))


def inference(classifier, image):
    """Perform preprocessing on image followed by inference using classifier"""
    image = TRANSFORM(image)
    image = image.unsqueeze(0)
    output = classifier(image)
    class_index = torch.argmax(output, dim=1)
    probability = torch.index_select(output, 1, class_index)
    return {"predicted_class": CLASSES[class_index], "probability": probability}


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
    load_model_weights(vgg16, os.path.join(MODEL_PATH, MODEL_B))
    return vgg16


def instantiate_net():
    """Instantiate a custom classifier object"""
    net_classifier = Net()
    load_model_weights(net_classifier, os.path.join(MODEL_PATH, MODEL_A))
    return net_classifier


if __name__ == "__main__":
    net_classifier = instantiate_net()
    vgg_classifier = instantiate_vgg16()
