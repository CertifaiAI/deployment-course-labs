# Exercise
You have gone through multiple examples of deploying model in Android application.\
Now you are required to build an application.

## Problem Statement
In the YOLOv5 example, you have learnt how to create an object detection application using YOLOv5. However,
the live view prediction of YOLOv5 is not fast enough.

In this lab, you are required to create an application to perform object detection that predicts 
[COCO dataset](https://cocodataset.org/#home). It is required to have at least 8 fps for live view prediction.

## Guiding Step
1. Main activity that act as the starting menu.

2. An activity with live view prediction, able to capture image, and show the bounding boxes instantly.

3. A light-weight object detection model that can support at least 8 fps for live view prediction.

4. Preprocessing class to convert images to tensor.

5. Class to draw bounding boxes on images.

