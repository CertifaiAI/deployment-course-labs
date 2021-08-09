from pydantic import BaseModel


class ResponseDataModel(BaseModel):
    """Constructor for data transfer object"""

    filename: str
    content_type: str
    likely_class: str
