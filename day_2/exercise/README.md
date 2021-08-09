# Lab Exercises

Here are three exercises designed to strengthen your understanding and mastery of the tools and frameworks that we have covered in `Day_2`.

## 1. Image Classifier Application using FastAPI

This exercise entails you to build a localhost API of image classifier which is accessible by other people/program for model predictions. You may choose any dataset or model that you are interested in to work on for this exercise. 

The application needs to at least fulfill the following requirements:
- `http://localhost:<PORT>` is accessible for POST/PUT request
- Application able to return appropriate server response (prediction results) to the caller
- `http://localhost:<PORT>/docs` is accessible for API documentation and trials

## 2. Image Classifier Application using FastAPI and Docker

This exercise entails you to build a containerized application of image classifier which is accessible by other people/program for model predictions. You may choose any dataset or model that you are interested in to work on for this exercise. 

The application needs to at least fulfill the following requirements:
- Application can be launched using Docker in the form of a container
- `http://localhost:<PORT>` is accessible for POST/PUT request
- Application able to return appropriate server response (prediction results) to the caller
- `http://localhost:<PORT>/docs` is accessible for API documentation and trials

## 3. Object Detector Application using FastAPI and Docker

This exercise entails you to build a containerized application of object detector which is accessible by other people/program for model predictions. You may choose any dataset or model that you are interested in to work on for this exercise. 

The application needs to at least fulfill the following requirements:
- Application is configured and launched using docker-compose in the form of a container
- `http://localhost:<PORT>` is accessible for POST/PUT request
- Application able to return appropriate server response (prediction results) to the caller
- `http://localhost:<PORT>/docs` is accessible for API documentation and trials

## Guiding Step

As a reminder, the typical workflow of a deployed model consists of the following 6 steps. Make sure that you have properly handled them in your application.

1. Validate the request
2. Gather additional data (context)
3. Preprocess data
4. Run the model
5. Postprocess the results
6. Return a result

## Tips

If you have no idea on where to start, `PyTorch` provides some handy dataset and models:

Data:
- [MNIST](https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html)
- [COCO](https://pytorch.org/vision/0.8/datasets.html#coco)

Model:
- [VGG nets](https://pytorch.org/hub/pytorch_vision_vgg/)
- [YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/)

