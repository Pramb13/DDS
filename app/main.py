from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import QueryRequest, QueryResponse
from app.services.pinecone_service import PineconeService
from app.services.huggingface_service import generate_response  # ✅ Updated

app = FastAPI(title="Driver Drowsiness API")

# CORS settings (adjust origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL on production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone service
pinecone_service = PineconeService()

@app.get("/")
async def root():
    return {"message": "🚗 Driver Drowsiness API with Hugging Face is running"}

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        # Get context from Pinecone
        pinecone_result = await pinecone_service.query_documents(request.query)
        context, references = pinecone_service.extract_context(pinecone_result)

        # Generate reply using Hugging Face model
        response_text = await generate_response(context, request.query)

        return QueryResponse(
            response=response_text,
            success=True,
            references=references
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
