# Deploy Image Classifier Using FastAPI

This repository will showcase how to deploy a deep learning image classifier in the form of web application using FastAPI.

### Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Dependencies](#dependencies)
4. [Directory Structure](#directory-structure)
5. [Model Used](#model-used)
6. [Running the Web App](#running-the-web-app)
7. [Testing Endpoints](#testing-endpoints)
8. [References](#references)

## System Requirements

The guide and code for this repository was written and tested on: 

* Ubuntu 20.04 LTS
* Python version 3.8.5 

It might still work properly for other OS or Python version, but expect to modify some of the shell commands or even code provided to make this repository works.

## Prerequisites

This tutorial will focus on building a REST API to serve image classifier web app. A basic understanding of the following would be necessary to follow along this tutorial:

* `PyTorch` 

If you feel like you would require a refresher on these subjects, you can learn more about `Pytorch` [here](https://pytorch.org/).

## Dependencies

Make sure that the terminal is at this directory (`deploy_image_classifier`). You can create a virtual environment first for this repository by:

```sh
python3 -m venv venv
```

Next, activate the virtual environment by:

```sh
source venv/bin/activate
```

After that, install all the dependencies of this repository in the environment by:

```sh
pip install -r requirements.txt
```

## Directory Structure
The directory for this repository is structured in the following manner:

```
deploy_image_classification
|
|- app/                                 (contains code for web app)
|   |- artifacts/                       (contains serialized files, place your model here)
|   |- __init__.py 
|   |- classifier.py                    (model constructor)
|   |- config.py                        (configuration details)
|   |- main.py                          (main script to set up model and API)
|   |_ models.py                        (data model)
|
|- sample_data/                         (contains images for testing purpose)
|- README.md                            (you are reading this now)
|_ requirements.txt                     (dependencies)
```

## Model Used 

The model to be used here is downloadable from [here](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/fruit_classifier_state_dict.pt). Place it under `app/artifacts/` after downloading. This model is trained and serialized in `deployment-course-labs/day_1/`'s repository, you can take a look at the training notebook [here](https://github.com/CertifaiAI/deployment-course-labs/blob/main/day_1/image_classifier_demo.ipynb).

## Running the Web App

Before running any command, we have to first set the PYTHONPATH first due to how this directory is structured. The PYTHONPATH needs to point to the root directory of this web app, which is `deploy_image_classifier/`.

To set PYTHONPATH in Linux, do the following (assuming you are at `//deploy_image_classifier/`):

```sh
export PYTHONPATH="$PWD"
```

If you use Windows, do the following (assuming you are at `...\deploy_image_classifier\`):

```PowerShell
set PYTHONPATH=%cd%
```

NOTE: Configuring PYTHONPATH this way is a one-time setting and exiting the terminal will reset the PYTHONPATH to null value. But this is good enough for the purpose of this tutorial.

There are two ways to run the web app:

### 1. From the Command Line by Invoking Uvicorn 
After installing the dependencies, use the following command to run the app from project root directory.

```sh
uvicorn app.main:app
```

### 2. From the Command Line by Invoking Python
After installing the dependencies, use the following command to run the app from project root directory.

```sh
python3 app/main.py
```

The terminal should display logs mentioning the successful deployment of the web app. In the case that you wish to enable re-deployment of web app everytime you make modifications to the codebase, add a `--reload` tag for the `uvicorn` command, for example:

```sh
uvicorn app.main:app --reload
```

## Testing Endpoints
There are few ways that you can test the REST API endpoint, we will just demonstrate one of them here. Visit **localhost:8000/docs** from your favourite browser. You will be able to see a documentation of the API. To test the endpoint, click the green `POST` button, `Try it out!` button and upload an image file followed by clicking the `Execute` button. You should be able to get response from the server.

If you would like to modify the code and have the browser reflects the changes, do the following:

```sh
uvicorn app.main:app --reload
```

## References:
- [Image Classification API](https://github.com/jabertuhin/image-classification-api)