from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.drowsiness_model import predict_drowsiness
from rag.rag_engine import query_logs

app = FastAPI()

# CORS Middleware to allow frontend access (e.g., from Streamlit Cloud)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic model for POST request body
class QueryInput(BaseModel):
    question: str

@app.get("/predict_drowsiness")
async def get_drowsiness_status():
    result = predict_drowsiness()
    return {"status": result}

@app.post("/query")
async def query_data(input_data: QueryInput):
    answer = query_logs(input_data.question)
    return {"answer": answer}
