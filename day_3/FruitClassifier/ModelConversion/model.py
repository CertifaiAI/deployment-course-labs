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
Model class for fruit classifier neural network
prepared by: YCCertifai
"""

import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # input shape is (1, 3, 150, 150)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(18496, 120)  # Note that the input of this layers is depending on your input image sizes
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)     # output shape is 3

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
