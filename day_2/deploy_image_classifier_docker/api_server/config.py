import torchvision.transforms as transforms
import os

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)

NUM_CLASSES = 4
CLASSES = ["COVID-19", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral"]
FILENAME = "xray_classifier_state_dict.pt"
FILE_PATH = os.path.realpath(__file__)
MODEL_WEIGHT_PATH = os.path.join(os.path.dirname(FILE_PATH), "artifacts", FILENAME)
