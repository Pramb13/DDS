import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

async def generate_response(context: str, query: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"].split("Answer:")[-1].strip()
    except Exception as e:
        print(f"Hugging Face Error: {e}")
        return "Sorry, I couldn't generate a response."
