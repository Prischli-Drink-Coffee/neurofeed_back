from pydantic import BaseModel


class Vebinar(BaseModel):
    name: str


class Feed(BaseModel):
    feedback: str
    topic: int
    useful: bool
    emotion: int
    keypoint: str

class User(BaseModel):
    username: str
    hashed_pass: str
    