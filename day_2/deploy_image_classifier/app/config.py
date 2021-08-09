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
FILE_PATH = os.path.realpath(__file__)
MODEL_WEIGHT_PATH = os.path.join(os.path.dirname(FILE_PATH), "artifacts", FILENAME)
print(MODEL_WEIGHT_PATH)
