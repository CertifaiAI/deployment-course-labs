import random
import cv2
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
import io

from app.config import COCO_INSTANCE_CATEGORY_NAMES

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def extract_results(inference_results: list, threshold: float):
    """Extract and filter classes and bounding box coordinates based on threshold"""
    inference_classes = [
        COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(inference_results[0]["labels"])
    ]
    inference_boxes = [
        [(float(i[0]), float(i[1])), (float(i[2]), float(i[3]))]
        for i in list(inference_results[0]["boxes"])
    ]
    inference_scores = list(inference_results[0]["scores"].detach().numpy())
    filtered_inferences_indices = [
        inference_scores.index(inference_score)
        for inference_score in inference_scores
        if inference_score > threshold
    ]
    filtered_inference_boxes = [
        inference_boxes[index] for index in filtered_inferences_indices
    ]
    filtered_inference_classes = [
        inference_classes[index] for index in filtered_inferences_indices
    ]
    return {
        "boxes": filtered_inference_boxes,
        "classes": filtered_inference_classes,
    }


def draw_bounding_boxes(
    image: Image,
    boxes: list,
    classes: list,
    rect_thickness=4,
    text_size=1,
    text_thickness=2,
):
    """Draw bounding box on original image based on inference results."""
    image_array = np.asarray(image)
    # Initialize random colors for different classes
    class_colors = {}
    for clas in classes:
        class_colors[clas] = [random.randint(0, 255) for _ in range(3)]

    for i in range(len(boxes)):
        cv2.rectangle(
            image_array,
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            (int(boxes[i][1][0]), int(boxes[i][1][1])),
            color=class_colors[classes[i]],
            thickness=rect_thickness,
        )
        cv2.putText(
            image_array,
            classes[i],
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            class_colors[classes[i]],
            thickness=text_thickness,
        )

    # save file in server
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(image_array)  # write image
    plt.xticks([])  # remove all xticks
    plt.yticks([])  # remove all yticks
    fig.savefig("sample_data/output/server_rendered.png")

    # convert nparray to base64 data format
    PIL_image = Image.fromarray(image_array)
    buff = io.BytesIO()
    PIL_image.save(buff, format="JPEG")
    base64str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64str
