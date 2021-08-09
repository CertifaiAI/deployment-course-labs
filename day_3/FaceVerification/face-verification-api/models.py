from pydantic import BaseModel


class FaceVerifyRequest(BaseModel):
    image1_string: str
    image2_string: str
