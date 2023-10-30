from pydantic import BaseModel

#   Class which describes the face side measurements
class ModelFile(BaseModel):
    img: str