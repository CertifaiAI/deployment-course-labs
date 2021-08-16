# Exercise: Train and Deploy a Drift Detector for X-Ray Classifier

As an exercise for training a drift detector, please 
- Train a drift detector for the X-ray classifier trained in Day 1 using the same set of training dataset.
-  Deploy the X-Ray classifier drift detector as a REST API to be queried from.

This is the suggested architecture:

![drift_detector_architecture](https://user-images.githubusercontent.com/53246500/125327367-9ecf5a00-e375-11eb-9c1d-ae4b20fde1be.png)

You may use any database of your choice and design the drift detector REST API according to the suggested architecture. However it is also possible that you come up with your own architectural design. 

Essentially, the objective of this exercise is for you to visualise 
- The items to be logged to perform drift detection
- The way to deploy a drift detector

In the suggested architecture, we are making use of the microservice architecture to deploy our model, log database and drift detector. It is advisable that you make use of a container orchestration tool to spin up all of the containers. 

## Suggested Tools
- [Fast API](https://fastapi.tiangolo.com/)
- [Docker](https://docs.docker.com/)
- [Docker-compose](https://docs.docker.com/compose/)
- [MongoDB](https://www.mongodb.com/)
