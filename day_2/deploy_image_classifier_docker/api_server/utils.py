import io
import base64
from PIL import Image
import os


def base64_str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode("utf-8")
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    image = Image.open(bytesObj)
    return image


if __name__ == "__main__":
    FILENAME = "Chest.jpeg"
    FILE_DIRECTORY = "COVID-19/"
    DATA_DIRECTORY = os.environ["PYTHONPATH"] + "/sample_data/"
    DECODED_FILENAME = FILENAME.split(".")[0] + "-decoded.jpeg"
    with open(DATA_DIRECTORY + FILE_DIRECTORY + FILENAME, "rb") as image:
        base64str = base64.b64encode(image.read()).decode("utf-8")
    image_PIL = base64_str_to_PILImage(base64str)
    image_PIL.save(DATA_DIRECTORY + FILE_DIRECTORY + DECODED_FILENAME)
