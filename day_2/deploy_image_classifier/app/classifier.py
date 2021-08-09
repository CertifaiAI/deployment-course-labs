from app.config import *
import torch
from torch import nn
import torch.nn.functional as F


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


class ImageClassifier:
    """Blueprint/Construct for image classifier"""

    def __init__(self):
        self.classifier = Net()
        self.classifier.load_state_dict(
            torch.load(MODEL_WEIGHT_PATH, map_location=torch.device("cpu"))
        )

    def predict(self, image):
        image = TRANSFORM(image)
        image = image.unsqueeze(0)
        output = self.classifier(image)
        class_index = torch.argmax(output, dim=1)

        return CLASSES[class_index]


if __name__ == "__main__":
    classifier = ImageClassifier()
    print(classifier.classifier)
