from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from models.drowsiness_model import predict_drowsiness
from rag.rag_engine import query_logs

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict_drowsiness")
async def get_drowsiness_status():
    result = predict_drowsiness()
    return {"status": result}

@app.post("/query")
async def query_data(request: Request):
    body = await request.json()
    question = body.get("question", "")
    answer = query_logs(question)
    return {"answer": answer}
