from fastapi import FastAPI, HTTPException

from app.rag.graph import ask_question
from app.schemas import ChatRequest, ChatResponse


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return ask_question(request.query)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )