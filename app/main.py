from agentic_rag import app as brain
from agentic_rag import rebuild_vectorstore

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler


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

def ensure_memory_initialized(thread_id: str):
    """Runs only ONCE per thread_id."""
    # Try to load memory by making a dummy read
    try:
        _ = brain.get_state({"configurable": {"thread_id": thread_id}})
        return  # Memory exists → do nothing
    except:
        pass  # No memory → initialize it

    # Initialize memory ONLY once
    brain.invoke(
        {
            "question": "Conversation will be started now.",
            "history": []
        },
        config={"configurable": {"thread_id": thread_id}}
    )


@app.post("/ask_agent")
def ask_agent(query:Query):
    ensure_memory_initialized(query.thread_id)

    result = brain.invoke(
        {"question": query.question},
        config={"configurable": {"thread_id": query.thread_id}}
    )

    return {
        "answer": result.get("answer", ""),
        "needs_retrieval": result.get("needs_retrieval", False),
        "documents": [doc.page_content for doc in result.get("documents", [])],
        "history": result.get("history", [])
    }


scheduler = BackgroundScheduler()

@scheduler.scheduled_job("interval", minutes=30)
def refresh_news_job():
    print("Refreshing news index...")
    rebuild_vectorstore()   # Re-scrape + rebuild FAISS inside agentic_rag.py
    print("News vectorstore updated.")

scheduler.start()