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
import os
import time
from pathlib import Path
import wget
import zipfile
import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

def download_model(source, target, filename):
    """ Download model from source to target directory """
    if not os.path.exists(target):
        os.mkdir(target) 
    target_file = str(Path(target).joinpath(filename))
    if os.path.exists(target_file):
        print('model already exists, skipping download')
        return
    print("Downloading from {} to {}".format(source, target))
    wget.download(source, target_file)  
    print("\nDone!")


def download_dataset(source, target, filename):
    if not os.path.exists(target):
        os.mkdir(target) 
    target_file = str(Path(target).joinpath(filename))
    if os.path.exists(target_file):
        print('data already exists, skipping download')
        return
    print("Downloading from {} to {}".format(source, target))
    wget.download(source, target_file)  
    print("\nDone!")
    print('Unzipping {}'.format(target_file))
    zipr = zipfile.ZipFile(target_file)
    zipr.extractall(target)
    zipr.close() 
    print('Done!')


def load_model_state_dict(model, model_state_dict_path):
    """
    A function to load model from specified state dict path location.

    Args:
        model_path (String): state dict path location

    Returns:
        Loaded model 
    """
    state_dict = torch.load(model_state_dict_path)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model


transform_composed = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_dataset(dataset_rootdir):
    """ Returns a dataloader given the dataset_rootdir"""
    dataset = datasets.ImageFolder(root=dataset_rootdir, transform=transform_composed)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return dataloader


def load_image(image_path):
    """ Returns an image given image_path"""
    image = Image.open(image_path)
    image = transform_composed(image)
    return image.unsqueeze(0)


def compute_accuracy(model, loader):
    model.eval()
    running_accuracy = 0.0

    for images, y_true in loader:

        # predict
        output = model(images)

        # calculate accuracy
        y_pred = torch.max(output, 1)[1]

        # adding accuracy per batch to running accuracy
        running_accuracy += accuracy_score(y_true, y_pred)

    overall_accuracy = running_accuracy / len(loader)

    return overall_accuracy


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, ' \t', 'Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size


def print_latency_of_model(model, inference_image, label=""):
    start_time = time.time()
    prediction = model(inference_image)
    end_time = time.time()
    inference_time = end_time - start_time
    print("model: ", label, ' \t', "prediction time: " + str(inference_time) + "s")
    return inference_time


def print_accuracy_of_model(model, testLoader, label=""):
    accuracy = compute_accuracy(model, testLoader)
    print("model: ", label, ' \t', "Test Accuracy: {0:.2f}".format(accuracy))
    return accuracy


def compare_performance(model_fp32, model_int8, label_model_fp32, label_model_int8, inference_image, test_dataloader):
    # 1. compare the model sizes
    print("Comparing size of models")
    original_size = print_size_of_model(model_fp32, label_model_fp32)
    quantised_size = print_size_of_model(model_int8, label_model_int8)
    print("{0:.2f} times smaller".format(original_size / quantised_size))

    # 2. compare latency of models
    print("\nComparing latency of models")
    print_latency_of_model(model_fp32, inference_image, label_model_fp32)
    print_latency_of_model(model_int8, inference_image, label_model_int8)

    # 3. compare accuracy of models
    print("\nComparing accuracy of models")
    print_accuracy_of_model(model_fp32, test_dataloader, label_model_fp32)
    print_accuracy_of_model(model_int8, test_dataloader, label_model_int8)