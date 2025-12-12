from app.agentic_rag import app as brain
from app.agentic_rag import rebuild_vectorstore

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import os
from dotenv import load_dotenv
load_dotenv()

import requests
from fastapi import Request

VERIFY_TOKEN = "flashFeedAgent"    # anything you want
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")





app = FastAPI(
    title="flashFeedAgent",
    description="Multi-agent RAG pipeline with LangGraph + News Scraper",
    version="1.0.0"
)

@app.get("/webhook")
def verify(request: Request):
    params = request.query_params
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return int(params.get("hub.challenge"))
    return "Verification failed"


def send_whatsapp_message(to, text):
    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": text
        }
    }

    print("Sending payload:", payload)

    response = requests.post(url, headers=headers, json=payload)
    print("Response:", response.status_code, response.text)

    return response.json()






@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    body = await request.json()

    try:
        message = body["entry"][0]["changes"][0]["value"]["messages"][0]
        user_msg = message["text"]["body"]
        user_id = message["from"]   # phone number
    except:
        return {"status": "ignored"}

    ensure_memory_initialized(user_id)

    # Call your Agentic RAG brain
    result = brain.invoke(
        {"question": user_msg,"history":[]},
        config={"configurable": {"thread_id": user_id}}
    )

    answer = result.get("answer", "Sorry, I couldn't process that.")

    send_whatsapp_message(user_id, answer)

    return {"status": "sent"}





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

