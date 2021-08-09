from torchvision import models
from torch import nn
import torch

from api_server.config import CLASSES, NUM_CLASSES, MODEL_WEIGHT_PATH, TRANSFORM


class SqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, NUM_CLASSES, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(13, 1),
        )
        print(MODEL_WEIGHT_PATH)
        self.model.load_state_dict(
            torch.load(MODEL_WEIGHT_PATH, map_location=torch.device("cpu"))
        )

    def predict(self, image):
        image = TRANSFORM(image)
        image = image.unsqueeze(0)
        output = self.model(image)
        class_index = torch.argmax(output, dim=1)

        return CLASSES[class_index]


if __name__ == "__main__":
    model = SqueezeNet()
    print(model)
