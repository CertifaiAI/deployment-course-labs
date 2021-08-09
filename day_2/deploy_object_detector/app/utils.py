import base64
import io
from PIL import Image
from torchvision import transforms


def bytes_to_base64(input: bytes):
    return base64.b64encode(input).decode("utf-8")


def base64str_to_PILImage(input_data: base64):
    """Convert base64str to Image"""

    base64_image_bytes = input_data.encode("utf-8")
    base64_bytes = base64.b64decode(base64_image_bytes)
    bytes_obj = io.BytesIO(base64_bytes)
    image = Image.open(bytes_obj)
    return image


def preprocessing(input_data: Image):
    """Convert image to tensor."""

    transformed_image = transforms.Compose([transforms.ToTensor()])(input_data)
    return transformed_image
