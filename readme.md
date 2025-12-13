# FlashFeedAgent 📰🤖  
**Agentic RAG–powered WhatsApp News Assistant**

FlashFeedAgent is an **Agentic RAG (Retrieval-Augmented Generation) system** that delivers **real-time news updates and contextual answers directly on WhatsApp**.  
It uses **LangGraph for agent orchestration**, **FastAPI for backend services**, **FAISS for vector retrieval**, and **WhatsApp Cloud API (Meta)** for messaging.

The system allows users to ask natural-language questions about current news and receive accurate, up-to-date responses via WhatsApp chat.



## 🚀 Key Features

- **WhatsApp-based interface** (no app, no login)
- **Agentic RAG pipeline** using LangGraph
- **Dynamic retrieval decision** (agent decides when to fetch news)
- **Web-scraped live news ingestion (refreshed every 30 mins)**
- **FAISS vector database** for semantic search
- **Per-user conversational memory** using phone number as `thread_id`
- **Webhook-driven real-time messaging**
- **FastAPI backend (production-ready)**


## 🧠 System Architecture

```markdown
User (WhatsApp)
     ↓
WhatsApp Cloud API (Meta)
    ↓  (Webhook POST)
FastAPI Backend (/webhook)
    ↓
LangGraph Agentic RAG
├── Retrieval Decision Agent
├── News Retrieval Agent (FAISS)
├── Answer Generation Agent (LLM)
└── Memory Update Agent
    ↓
WhatsApp Cloud API (Send Message)
    ↓
  User

```

## 🧩 Agentic RAG Design (LangGraph)


| Agent | Responsibility |
|------|----------------|
| Retrieval Decision Agent | Determines if live news retrieval is required |
| Retrieval Agent | Fetches relevant news chunks from FAISS |
| Answer Generation Agent | Generates final response using context |
| Memory Agent | Updates per-user conversational history |

Each agent is an independent node in a LangGraph state machine.

## 💬 WhatsApp Integration (Cloud API)

- Uses **WhatsApp Cloud API** (Meta)
- Messages are received via **webhooks**
- Replies are sent using the `/messages` API
- User phone number is used as:
  - Unique user identifier
  - LangGraph `thread_id`
  - WhatsApp reply destination

Follow-up questions are supported via conversational memory.

### Why Webhooks?
WhatsApp pushes incoming messages to your backend automatically — no polling required.


## 🔐 Authentication & Tokens

Environment variables required:


- GROQ_API_KEY=your_llm_api_key
- WHATSAPP_TOKEN=meta_access_token
- VERIFY_TOKEN=webhook_verify_token
- PHONE_NUMBER_ID=whatsapp_phone_number_id

## 🛠 Tech Stack

- **Backend:** FastAPI
- **Agent Framework:** LangGraph
- **LLM Provider:** Groq (LLaMA)
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace (gte-small)
- **Messaging:** WhatsApp Cloud API
- **Scraping:** BeautifulSoup + RecursiveUrlLoader
- **Tunnel (Dev):** ngrok


## 📦 Installation & Local Setup

```bash
git clone https://github.com/a-anuj/flashFeedAgent.git
cd flashFeedAgent
pip install -r requirements.txt
```

### Run Locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Expose webhook (dev)
```bash
ngrok http 8000
```


