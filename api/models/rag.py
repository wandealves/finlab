from pydantic import BaseModel


class RagRequest(BaseModel):
    query: str
    limit: int = 3


class RagResponse(BaseModel):
    query: str
    answer: str
    metadata: list[dict]
