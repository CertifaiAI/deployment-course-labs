# Deploy Object Detector Using Docker Compose

This repository will showcase how to build and deploy a deep learning object detector using FastAPI and Docker Compose.

### Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Dependencies](#dependencies)
4. [Directory Structure](#directory-structure)
5. [Model Used](#model-used)
6. [Running the Web App](#running-the-web-app)
7. [Dockerization](#dockerization)
8. [Docker Compose](#docker-compose)
9. [Testing Endpoints](#testing-endpoints)
10. [References](#references)

## System Requirements

The guide and code for this repository was written and tested on: 
* Ubuntu 20.04 LTS
* Python version 3.8.5 

It might still work properly for other OS or Python version, but expect to modify some of the shell commands or even code provided to make this repository works.

## Prerequisites

This tutorial will focus on dockerizing an object detector web app served using FastAPI. A basic understanding of the followings would be necessary to follow along this tutorial:

* `PyTorch` 
* `FastAPI`
* `Docker`

If you feel like you would require a refresher on these subjects, you can learn more about `Pytorch` [here](https://pytorch.org/), `FastAPI` [here](https://fastapi.tiangolo.com/tutorial/) and `Docker` [here](https://docs.docker.com/).

## Dependencies

Make sure that the terminal is at this directory (`deploy_object_detector`). You can create a virtual environment first for this repository by:

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

The following is performed to set up the PYTHONPATH for the `app/` directory so that the packages inside it can be imported.

```sh
pip install -e .
```

This command will then install the `app/` folder as a Python module in the activated virtual environment, which allowed for import of any files inside the module to be possible from anywhere.

One thing to note is that this virtual environment is only used for local development purpose. Once you have tested that the web app is able to be deployed locally, we can proceed on with the dockerization process of the web app.

## Directory Structure

The directory for this repository is structured in the following manner:

```
deploy_object_detector
|
|- app/                                 (contains code to serve API)
|   |- static/                          (contains static files, you can place your styling files here)
|   |- templates/                       (contains static HTML files)
|   |- __init__.py 
|   |- config.py                        (configuration details)
|   |- detector.py                      (code for deep learning model)
|   |- main.py                          (main script to set up model and API)
|   |- models.py                        (code for data models)
|   |_ utils.py                         (code for various utilities)
|
|- sample_data/                         (contains images for testing purpose)
|   |_ output/                          (directory to save output images by web app)
|- tests/                               (contains test cases)
|- docker-compose.yml                   (configuration for Docker application)
|- Dockerfile                           (contains instructions on building Docker image)
|- README.md                            (you are reading this now)
|- requirements.txt                     (dependencies)
|_ setup.py                             (set up app directory to Python module) 
```

## Model Used 

The model to be used here is a pretrained model from [TORCH.HUB](https://pytorch.org/docs/stable/hub.html). The code will inspect whether the model exists everytime when we run the model, and it will download it from `torch.hub` if the model does not exist. So, do expect some waiting time for the first run.

## Running the Web App

There are two ways to run the web app locally (not using Docker container):

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

The terminal should display logs mentioning the successful deployment of the web app. 

## Dockerization

Run the following to build an image from the `Dockerfile`:

```sh
docker build -t <DOCKER_ID>/object-detector .
```

Test the functionality of the Docker image by running a container and exposing a port in the container:

```sh
docker run -p <SERVER_PORT>:<CONTAINER_PORT> <DOCKER_ID>/object-detector
```

This should bring the container up and running. You can proceed with testing the API endpoints by following the instructions [here](#testing-endpoints).

## Docker Compose

We will demonstrate an over-simplified deployment option using Docker Compose. Docker Compose is a tool in the Docker family used to define and run multiple containers at the same time. A YAML file is necessary to configure the application, serving sort of like a blueprint for building the entire application. Generally, it consists of the following three step process:

1. Define the application's environment with a `Dockerfile` for reproducibility regardless of environment [covered]

2. Create a `docker-compose.yml` file to configure all the services which make up the application (to run them together in an isolated environment) [created]

3. Run the following to spin up the application:
```sh
docker-compose up
```

To spin down the application, run the following from another terminal:
```sh
docker-compose down
```

Or alternatively, press `Ctrl+C` in the current terminal.

PS: For all `docker-compose` related commands, we have to run it in the directory containing the `docker-compose.yml` file as it will implicitly look for the file in the current directory (which in this case, is `deploy_object_detector/`).

If you want to read more on Docker Compose and its features, click [here](https://docs.docker.com/compose/#features). If you would like to know more on other useful Docker Compose commands, click [here](https://docs.docker.com/compose/cli-command/).

## Testing Endpoints

There are a few ways to test the endpoints:

1. Running the following in your terminal:

```sh
python ./tests/test_endpoint.py
```

2. Launching the web app at 
```http://localhost:<SERVER_PORT>/interface```

The resultant output will be stored at `sample_data/output/` folder for your viewing.

## References
- [Form Data](https://fastapi.tiangolo.com/tutorial/request-forms/)
- [Base64 to Image](https://codebeautify.org/base64-to-image-converter)
- [How to convert Python numpy array to base64 output](https://stackoverflow.com/questions/43310681/how-to-convert-python-numpy-array-to-base64-output)
- [Deployment could be easy — A Data Scientist’s Guide to deploy an Image detection FastAPI API using Amazon ec2](https://mlwhiz.com/blog/2020/08/08/deployment_fastapi/)
- [Request Files](https://fastapi.tiangolo.com/tutorial/request-files/#multiple-file-uploads)
- [Jinja template inheritance | Learning Flask Ep. 5](https://pythonise.com/series/learning-flask/jinja-template-inheritance)
- [ImportError: libGL.so.1: cannot open shared object file: No such file or directory](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)