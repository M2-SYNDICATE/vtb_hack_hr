from pydantic import BaseModel

class JobPosting(BaseModel):
    comment: str
    answer: bool