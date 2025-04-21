from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import QueryRequest, QueryResponse, DrowsinessResponse
from app.services.drowsiness_service import predict_drowsiness
from app.services.rag_service import query_logs

app = FastAPI(title="Driver Drowsiness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    return {"message": "Driver Drowsiness Detection API is running"}

@app.get("/predict_drowsiness", response_model=DrowsinessResponse)
async def get_drowsiness_status():
    try:
        status = predict_drowsiness()
        return DrowsinessResponse(status=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        answer = query_logs(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))