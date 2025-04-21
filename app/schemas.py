from pydantic import BaseModel

class DrowsinessResponse(BaseModel):
    status: str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str