from agentic_rag import app as brain

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="flashFeedAgent",
    description="Multi-agent RAG pipeline with LangGraph + News Scraper",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class Query(BaseModel):
    question: str
    thread_id: str = "default-session"   # Used for persistent memory


# Health Check
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Backend running"}