# Deployment Strategies Using A/B Testing

This repository will simulate how to deploy a new deep learning image classifier using a crude version of A/B testing strategy.

### Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Dependencies](#dependencies)
4. [Directory Structure](#directory-structure)
5. [A/B Testing](#ab-testing)
5. [Model Used](#model-used)
6. [Running the Containers](#running-the-containers)
7. [Testing Endpoints](#testing-endpoints)
8. [References](#references)

## System Requirements

The guide and code for this repository was written and tested on: 

* Ubuntu 20.04 LTS
* Python version 3.8.5 

It might still work properly for other OS or Python version, but expect to modify some of the shell commands or even code provided to make this repository works.

## Prerequisites

This tutorial will focus on demonstrating on how to deploy two different models and route traffic accordingly using A/B testing strategy. A basic understanding of the followings would be necessary to follow along this tutorial:

* `PyTorch` 
* `FastAPI`
* `Docker`

If you feel like you would require a refresher on these subjects, you can learn more about `Pytorch` [here](https://pytorch.org/), `FastAPI` [here](https://fastapi.tiangolo.com/tutorial/) and `Docker` [here](https://docs.docker.com/).

## Dependencies

The most advisable way for testing the application individually is by building a Docker image from the respective `Dockerfile` located in the application directory and run the container. This can be achieved by:

```sh
docker build -t <IMAGE_TAG> .
docker run -p <CONTAINER_PORT>:<SERVER_PORT> <IMAGE_TAG>
```

To install local dependencies for each individual application and test using local machine, you can create a virtual environment for that application by:

```sh
cd <APPLICATION_FOLDER_PATH/>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Directory Structure
The directory for this repository is structured in the following manner:

```
AB_testing
|
|- app_custom/                          (contains application code and artifacts for custom model)
|- app_vgg16/                           (contains application code and artifacts for VGG16 model)
|- nginx/                               (contains configuration code for NGINX)
|- sample_data/                         (contains images for testing purpose)
|- docker-compose.yml                   (contains configuration code for docker-compose)
|_ README.md                            (you are reading this now)
```

## A/B Testing

There are several strategies to deploy and test your model in production. The specific one that is shown here is A/B testing. To be concise, this is an overly simplified scenario whereby we assume that we already have a model in production, which is served by a server named `app_custom` on port 8009. We have decided to improve the application by rolling out a newly trained model by applying transfer learning on VGG16 model. As with anything in production, we have to test it first before replacing the old model with the new one.

Although A/B testing is a form of empirical experiment whereby a specific set of criteria needs to be set and met beforehand (for example, all requests made by males are routed to server A and all requests made by females are routed to server B),we will not be doing that to keep the simplicity of this demonstration. The main purpose of this demonstration is to emphasize on the idea that new models should be tested before being rolled out into production and and how we can go about doing so.

The new model will be served by another server named `app_vgg16` on port 8008. NGINX is used as the reverse proxy to route traffic to both servers using round robin strategy.

## Model Used

Please run the following command at terminal at `AB_testing/` directory to download the serialized models:

```py
python download_models.py
```

Alternatively, you can download and place the two models manually in the respective directory:

1. The model to be used by `app_custom` server is downloadable from [here](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/fruit_classifier_state_dict.pt). Place it under `app_custom/app/artifacts/` after downloading. 

2. The model to be used by `app_vgg16` server is downloadable from [here](https://s3.eu-central-1.wasabisys.com/certifai/deployment-training-labs/models/transfer_learning_model.pt). Place it under `app_vgg16/app/artifacts/` after downloading. 

Both models is trained and serialized in `deployment-course-labs/day_1/`'s repository, you can take a look at the training notebook [here](https://github.com/CertifaiAI/deployment-course-labs/blob/main/day_1/image_classifier_demo.ipynb).

## Running the Containers

The simplest way of running the containers (`app_custom`, `app_vgg16` and `nginx`) at one-go is by using `docker-compose`. Ensure that you are running the following from `AB_testing/` directory:

```sh
docker-compose up --build
```

## Testing Endpoints

To visit the application, go to **localhost** in any web browser. NGINX will then route the request to any of the two servers in sequential order. Upload the test image and click `Submit` button to check the returned result, and you should see that different results will be returned depending on which server that the request is routed to (hence different model). 

To check which server is handling the request, inspect the logs generated by Docker in the terminal that you ran the `docker-compose` command.

## References:
- [Deploying Machine Learning model](https://medium.com/analytics-vidhya/deploying-machine-learning-model-f3af52068c1b)
- [Understanding Nginx HTTP Proxying, Load Balancing, Buffering, and Caching](https://www.digitalocean.com/community/tutorials/understanding-nginx-http-proxying-load-balancing-buffering-and-caching)
- [Argparse Tutorial](https://docs.python.org/3/howto/argparse.html)