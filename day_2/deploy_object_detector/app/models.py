from pydantic import BaseModel


class Base64str(BaseModel):
    base64str: str
    threshold: float
