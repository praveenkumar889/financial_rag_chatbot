from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from rag_core.engine import answer_question

# Create FastAPI app
app = FastAPI(title="Financial RAG API")

# Request schema
class QueryRequest(BaseModel):
    question: str

# Health endpoint
@app.get("/")
def health_check():
    return {"status": "RAG API is running"}

# Main RAG endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        response = answer_question(request.question)
        # Ensure we only return the answer string to match CLI behavior
        return {"answer": response.get("answer", "No answer found.")}
    except Exception as e:
        return {"error": str(e)}
