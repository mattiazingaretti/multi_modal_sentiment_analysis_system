from pydantic import BaseModel


class TestModelRequestDTO(BaseModel):
  text: str
  image_path: str