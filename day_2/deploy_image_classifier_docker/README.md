# Deploy Image Classifier Using Docker

This repository will showcase how to build and deploy a deep learning image classifier using FastAPI and Docker.

### Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Dependencies](#dependencies)
4. [Directory Structure](#directory-structure)
5. [Model Used](#model-used)
6. [Running the Web App](#running-the-web-app)
7. [Dockerization](#dockerization)
8. [Testing Endpoints](#testing-endpoints)
9. [References](#references)

## System Requirements

The guide and code for this repository was written and tested on: 
* Ubuntu 20.04 LTS
* Python version 3.8.5 

It might still work properly for other OS or Python version, but expect to modify some of the shell commands or even code provided to make this repository works.

## Prerequisites

This tutorial will focus on dockerizing an image classifier web app served using FastAPI. A basic understanding of the following would be necessary to follow along this tutorial:

* `PyTorch` 
* `FastAPI`

If you feel like you would require a refresher on these subjects, you can learn more about `Pytorch` [here](https://pytorch.org/) and `FastAPI` [here](https://fastapi.tiangolo.com/tutorial/).

## Dependencies

Make sure that the terminal is at this directory (`deploy_image_classifier_docker`). You can create a virtual environment first for this repository by:

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

Nonetheless, this virtual environment is only used for local development purpose. Once you have tested that the web app is able to be deployed locally, we can proceed on with the dockerization process of the web app.

## Directory Structure

The directory for this repository is structured in the following manner:

```
deploy_image_classifier_docker
|
|- api_server/                          (contains code to serve API)
|   |- artifacts/                       (contains serialized files, you can place your serialized model here)
|   |- __init__.py 
|   |- classifier.py                    (model constructor)
|   |- config.py                        (configuration details)
|   |- data_validation.py               (validate request data using Pydantic)
|   |- feature_extractor.py             (extract features)
|   |_ main.py                          (main script to set up model and API)
|
|- sample_data/                         (contains images for testing purpose)
|- tests/                               (contains test cases)
|- Dockerfile                           (contains instructions on building Docker image)
|- README.md                            (you are reading this now)
|_ requirements.txt                     (dependencies)
```

## Model Used 

The model to be used here is downloadable from [here](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/xray_classifier_state_dict.pt). Place it under `api_server/artifacts/` after downloading. This model is trained and serialized in `DIRECTORY HERE/`'s repository, you can take a look at the training notebook [here]().

## Running the Web App

Before running any command, we have to first set the PYTHONPATH first due to how this directory is structured. The PYTHONPATH needs to point to the root directory of this web app, which is `deploy_image_classifier_docker/`.

To set PYTHONPATH in Linux, do the following (assuming you are at `//deploy_image_classifier_docker/`):

```sh
export PYTHONPATH="$PWD"
```

If you use Windows, do the following (assuming you are at `...\deploy_image_classifier_docker\`):

```cmd
set PYTHONPATH=%cd%
```

NOTE: Configuring PYTHONPATH this way is a one-time setting and exiting the terminal will reset the PYTHONPATH to null value. But this is good enough for the purpose of this tutorial.

There are two ways to run the web app locally (not using Docker container):

### 1. From the Command Line by Invoking Uvicorn 
After installing the dependencies, use the following command to run the app from project root directory.

```sh
uvicorn api_server.main:app
```

### 2. From the Command Line by Invoking Python
After installing the dependencies, use the following command to run the app from project root directory.

```sh
python3 api_server/main.py
```

The terminal should display logs mentioning the successful deployment of the web app. 

## Dockerization

Docker is one of the most popular container service out there that greatly simplifies and accelerates a developer's workflow by enabling isolation of application from its environment. You can read more about Docker on its [official page](https://www.docker.com/why-docker).

In order to use Docker, we have to install and configure it into our system. Follow the guide provided [here](https://github.com/CertifaiAI/mldl-traininglabs/wiki/Setting-Up-Docker) to do so.

Having Docker now properly configured into our system, we can easily build an image from a `Dockerfile` with the following:

```sh
docker build -t <DOCKER_ID>/x-ray-classifier .
```

A basic Docker run command as below can be used to run the container based on the image that we have provided:

```sh
docker run <DOCKER_ID>/x-ray-classifier
```

However, note that it does not work as we have to open up a port from the running Docker container to connect to the port of the server. To achieve this, use the following:

```sh
docker run -p <SERVER_PORT>:<CONTAINER_PORT> <DOCKER_ID>/x-ray-classifier
```

This should bring the container up and running. You can proceed with testing the API endpoints by following the instructions [here](#testing-endpoints).

After you have done working with the container, we have to manually shut it down from another terminal as we will be having no access to STDIN of the terminal that is running Docker container now. To do this, open up another terminal and do the following:

```sh
docker ps
```

This will bring up a list of Docker containers that are running. Highlight the relevant container ID and copy it as we need it later.

```sh
docker kill <CONTAINER_ID>
```

Paste the copied container ID in the above command and run it into the terminal. This command will terminate the specified Docker container.

## Testing Endpoints

You can test the API endpoint in your local machine by running the following in your terminal:

```sh
python ./tests/test_predict_endpoint.py
```

Or you can also do a `curl` using image in `base64str` file format manually.

## References:
- [Docker Tutorial For Beginners - How To Containerize Python Applications](https://www.youtube.com/watch?v=bi0cKgmRuiA)
- [What is a Container?](https://www.docker.com/resources/what-container)
- [Deployment could be easy — A Data Scientist’s Guide to deploy an Image detection FastAPI API using Amazon ec2](https://mlwhiz.com/blog/2020/08/08/deployment_fastapi/)