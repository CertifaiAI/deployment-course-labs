import requests, json
import base64
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2


def send_request(image):
    """Send HTTP post request using specified image"""
    with open(image, "rb") as image_file:
        base64str = base64.b64encode(image_file.read()).decode("utf-8")
    payload = json.dumps({"base64str": base64str, "threshold": 0.5})
    response = requests.post("http://127.0.0.1:8000/predict", data=payload)
    print(response)
    return response.json()


def PILImage_to_cv2(img):
    """Convert PIL to cv2"""
    return np.asarray(img)


def draw_bounding_box(img, boxes, pred_cls, rect_th=2, text_size=1, text_th=2):
    """Draw bounding boxes on client-end using response received"""
    img = PILImage_to_cv2(img)
    class_color_dict = {}

    # initialize some random colors for each class for better looking bounding boxes
    for cat in pred_cls:
        class_color_dict[cat] = [random.randint(0, 255) for _ in range(3)]

    for i in range(len(boxes)):
        cv2.rectangle(
            img,
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            (int(boxes[i][1][0]), int(boxes[i][1][1])),
            color=class_color_dict[pred_cls[i]],
            thickness=rect_th,
        )
        cv2.putText(
            img,
            pred_cls[i],
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            class_color_dict[pred_cls[i]],
            thickness=text_th,
        )
    fig = plt.figure(figsize=(30, 20))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    fig.savefig("sample_data/output/client_rendered.png")


if __name__ == "__main__":
    data_dict = send_request("sample_data/dog_with_ball.jpg")
    img = Image.open("sample_data/dog_with_ball.jpg")
    draw_bounding_box(img, data_dict["boxes"], data_dict["classes"])
