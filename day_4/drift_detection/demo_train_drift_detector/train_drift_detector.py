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

import copy
import os
import torch
import torchdrift
from model import Net
from data_generator import load_dataset
from utils import download_dataset, download_model, load_model_state_dict
import pickle

# PATHS FOR DATA DOWNLOAD
DATA_DOWNLOAD_PATH = "https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/fruits_image_classification-20210604T123547Z-001.zip"
DATA_SAVE_PATH = "../../../resources/data"
DATA_ZIP_FILENAME = "fruits_image_classification.zip"

# PATHS OF DATASETS
TRAIN_DATASET_ROOTDIR = "../../../resources/data/fruits_image_classification/train"
TEST_DATASET_ROOTDIR = "../../../resources/data/fruits_image_classification/test"
DIRTYTEST_DATASET_ROOTDIR = "../../../resources/data/fruits_image_classification/dirty_test"

# PATHS FOR MODEL DOWNLOAD
MODEL_DOWNLOAD_PATH = 'https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/fruit_classifier_state_dict.pt'
MODEL_STATE_DICT_PATH = '../../../resources/model/'
FILENAME = 'fruits_image_classification.zip'

# PATH TO SAVE PICKLED DRIFT DETECTOR
GENERATED_MODEL_PATH = "../../../generated_model"

def main():
    ################ load data ################
    # download dataset
    download_dataset(DATA_DOWNLOAD_PATH, DATA_SAVE_PATH, DATA_ZIP_FILENAME)

    # load train data
    train_dataloader = load_dataset(TRAIN_DATASET_ROOTDIR)

    # load clean test data
    test_dataloader = load_dataset(TEST_DATASET_ROOTDIR)

    # load dirty test data
    dirtytest_dataloader = load_dataset(DIRTYTEST_DATASET_ROOTDIR)

    ################ load model ################
    download_model(MODEL_DOWNLOAD_PATH, MODEL_STATE_DICT_PATH, FILENAME)
    model = Net()
    model = load_model_state_dict(model, MODEL_STATE_DICT_PATH + FILENAME)

    # get only the feature extractor
    ################ only use the model to extract features ################
    feature_extractor = copy.deepcopy(model)

    # we do not need the fully connected layers
    # torch.nn.Identity is just a placeholder layer
    feature_extractor.fc1 = torch.nn.Identity()
    feature_extractor.fc2 = torch.nn.Identity()
    feature_extractor.fc3 = torch.nn.Identity()

    print("\nFeature extractor architecture: ")
    print(feature_extractor)

    # set feature extractor to eval mode
    feature_extractor.eval().to("cpu")
    for p in feature_extractor.parameters():
        p.requires_grad_(False)


    ################ train MMD drift detector ###################
    # initialise drift detector
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()

    # fit drift detector to our training data
    print("\nTraining MMD drift detector:")
    torchdrift.utils.fit(train_dataloader, feature_extractor, drift_detector, num_batches=1)

    ################ test the trained MMD drift detector ###################
    # test our drift detector on dirty test data
    inputs, _ = next(iter(dirtytest_dataloader))
    # inputs, _ = next(iter(train_dataloader)) # uncomment to see what result will you get if you test it on the training set
    # inputs, _ = next(iter(test_dataloader)) # uncomment to see what result will you get if you test it on the clean test set

    # extract features using the feature extractor defined earlier
    features = feature_extractor(inputs)

    # compute test score (loss)
    score = drift_detector(features)

    # Note that for the p - value lower means "more likely that things have drifted"
    # H0 = no drift; Ha = data drifted
    # hence if our p-value is small, we will reject H0 which hypotheses that the data have not drifted
    # and accept the alternative hypothesis (Ha) which hypotheses that the data have drifted
    p_val = drift_detector.compute_p_value(features)
    print("\nTest score: ", str(score.item()), "  p-value: ", str(p_val.item()))

    # serialise the drift detector in a pickle
    if not os.path.exists(GENERATED_MODEL_PATH):
        os.mkdir(GENERATED_MODEL_PATH)
    pickle.dump(drift_detector, open(GENERATED_MODEL_PATH + "/drift_detector.pkl", 'wb'))

if __name__ == '__main__':
    main()


