#!/usr/bin/env python
# coding: utf-8

# In[78]:


#pip install google-generativeai

import pandas as pd
pd.set_option('display.max_colwidth', None)


# In[79]:


import warnings
warnings.filterwarnings("ignore")


# In[80]:


import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class GroqLLM:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def __call__(self, query):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": str(query)}]
        )
        return response.choices[0].message.content

# Initialize like this
llm = GroqLLM(client, "llama-3.3-70b-versatile")


# In[81]:


import os
from typing import List,TypedDict
from langgraph.graph import StateGraph,END
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# In[82]:

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os

emb = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_TOKEN"),
    model_name="thenlper/gte-small"
)

# In[121]:


from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

news_sites = [
    "https://www.bbc.com/news",
    "https://www.bbc.com/sport",
    "https://www.bbc.com/",
    "https://www.bbc.com/business",
    "https://www.bbc.com/innovation",
    "https://www.bbc.com/culture",
    # "https://www.aljazeera.com/",
    "https://www.bbc.com/travel",
    "https://www.bbc.com/earth",
    "https://www.bbc.com/arts",
    "https://www.bbc.com/live",
    "https://indianexpress.com/section/india/",
    # "https://www.thehindu.com/news/",
    # "https://www.reuters.com/world/",
    # "https://www.thehindu.com/news/national/",
    # "https://www.thehindu.com/news/international/",
    # "https://www.thehindu.com/entertainment/movies/",
    # "https://www.thehindu.com/sport/",
    # "https://www.thehindu.com/data/",
    # "https://www.thehindu.com/sci-tech/health/",
    # "https://www.thehindu.com/opinion/",
    # "https://www.thehindu.com/sci-tech/science/",
    # "https://www.thehindu.com/business/",

]

all_docs = []

def extract_text(html):
    return Soup(html, "html.parser").get_text(separator="\n")

for url in news_sites:
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=1,              # Keep shallow for news (faster, avoids old archives)
        extractor=extract_text
    )

    docs = loader.load()
    all_docs.extend(docs)

print(f"Loaded total documents: {len(all_docs)}")


# In[122]:


document = all_docs
print(document)


# In[123]:


from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context. 
Also if any code snippet is available please provide it.
Always answer for more then 100 words. 
<context>
{context}
</context>
Question : {input}""")


# In[124]:


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# In[125]:


splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500
)

docs = splitter.split_documents(document)


# In[126]:


vectorstore = FAISS.from_documents(docs,embedding=emb)


# In[127]:


retriever = vectorstore.as_retriever(k=5)


# In[128]:


# context_docs = retriever.invoke("Give to me in detail about how to use agents in langchain")
# context = format_docs(context_docs)

# final_prompt = prompt.format(context=context, input="Give to me in detail about how to use agents in langchain")

# response = model.generate_content(final_prompt)
# print(response.text)


# In[129]:


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer:str
    needs_retrieval: bool
    history:List


# In[130]:


def update_history(state:AgentState) -> AgentState:
    history = state["history"]
    history.append({
        "user":state["question"],
        "assistant":state["answer"]
    })
    return {**state,"history":history}


# In[131]:


def decide_retrieval(state: AgentState) -> AgentState:
    question = state["question"]

    response = llm(
        f"""
        Return ONLY True or False.

        True → The question REQUIRES checking the news knowledge base because it asks about:
        - recent events
        - current affairs
        - politics, sports, economy, technology updates
        - breaking news or anything time-sensitive
        - information that changes daily and must come from scraped news articles

        False → The question is general knowledge OR does not depend on recent news.

        Do NOT answer based on whether the LLM knows the answer.
        Decide ONLY based on whether the latest news articles are required.

        Question: {question}
        """
        )

    cleaned = response.strip().lower()
    needs_retrieval = cleaned.startswith("t")

    return {**state, "needs_retrieval": needs_retrieval}


# In[132]:


def retrieve_documents(state:AgentState) -> AgentState:
    question = state['question']

    documents = retriever.invoke(question)

    return {**state,"documents":documents}


# In[133]:


def generate_answer(state:AgentState) -> AgentState:
    question = state['question']
    documents = state.get("documents",[])
    history = state["history"]

    if documents:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"""
                    Based on the following context, answer the question : 
                    Context : {context}
                    Question : {question}
                    Conversational History : {history}
                    Answer : 
                    """
    else:
        prompt = f"""Answer the following question : 
                    Conversational History : {history}
                    Question : {question}"""
    response = llm(prompt)
    answer = response

    return {**state,"answer":answer}


# In[134]:


def should_retrieve(state:AgentState) -> AgentState:
    if state["needs_retrieval"]:
        return "retrieve"
    else:
        return "generate"


# In[135]:


workflow = StateGraph(AgentState)

workflow.add_node("decide",decide_retrieval)
workflow.add_node("retrieve",retrieve_documents)
workflow.add_node("generate",generate_answer)
workflow.add_node("history",update_history)


# In[136]:


workflow.set_entry_point("decide")

workflow.add_conditional_edges(
    "decide",
    should_retrieve,
    {
        "retrieve" : "retrieve",
        "generate" : "generate"
    }

)

workflow.add_edge("retrieve","generate")
workflow.add_edge("generate","history")
workflow.add_edge("history",END)

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)



# In[137]:


def ask_question_start(question:str):
    result = app.invoke(
        {"question":question,"history":[]},
        config={"configurable":{"thread_id":"news-session-1"}}
        )
    return result


# In[138]:


def ask_question(question:str):
    result = app.invoke(
        {"question":question},
        config={"configurable":{"thread_id":"news-session-1"}}
        )
    return result


# In[139]:


ask_question_start("Conversation is going to start...")


# In[153]:


question = "What is the status of ashes now?"
result = ask_question(question)
answer = result


# In[154]:


print(answer['answer'])


# In[155]:


print(result["needs_retrieval"])


# In[156]:


print(result["documents"])


def rebuild_vectorstore():
    global retriever
    global vectorstore
    global docs
    global all_docs

    print("Scraping fresh news...")

    new_docs = []
    for url in news_sites:
        loader = RecursiveUrlLoader(url=url, max_depth=1, extractor=extract_text)
        new_docs.extend(loader.load())

    # Split docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = splitter.split_documents(new_docs)

    # Build FAISS
    vectorstore = FAISS.from_documents(docs, embedding=emb)
    retriever = vectorstore.as_retriever(k=5)

    print("News scraping + vectorstore rebuilt successfully.")

