# app.py
import os
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

from sqlalchemy import create_engine, Table, Column, Integer, String, ForeignKey, DateTime, Text, MetaData, select, insert, update, delete
from sqlalchemy.orm import sessionmaker

# LangChain imports (keep same as your environment expects)
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ------------------------
# App & CORS
# ------------------------
app = FastAPI(title="RAG Chat Backend (No Auth)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Config
# ------------------------
DATABASE_URL = "sqlite:///rag_chat_app.db"
API_KEY = os.getenv("API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ------------------------
# Don't-know messages
# ------------------------
DONT_KNOW = {
    "English": "I don't know about this.",
    "Hindi": "मुझे नहीं पता।",
    "Marathi": "मला माहित नाही."
}
ALLOWED_LANGS = set(DONT_KNOW.keys())

# ------------------------
# DB setup
# ------------------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String, unique=True, nullable=False),
    Column("password_hash", String, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow)
)

conversations = Table(
    "conversations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=True),  # no auth -> optional
    Column("title", String),
    Column("created_at", DateTime, default=datetime.utcnow)
)

messages = Table(
    "messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("conversation_id", Integer, ForeignKey("conversations.id")),
    Column("role", String),
    Column("content", Text),
    Column("timestamp", DateTime, default=datetime.utcnow)
)

prompt_answer = Table(
    "prompt_answer", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("prompt", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("timestamp", DateTime, default=datetime.utcnow)
)

metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ------------------------
# LangChain / embeddings / LLM
# ------------------------
os.environ["HF_TOKEN"] = HF_TOKEN or ""

# Initialize embeddings & LLM once
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

llm = ChatGroq(groq_api_key=API_KEY, model_name="llama-3.1-8b-instant")

# ------------------------
# In-memory conversation memory (session-store)
# ------------------------
memory_store = {}  # session_id (str) -> ChatMessageHistory

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

# ------------------------
# Helper: build_rag() rebuilds vectorstore from DB rows
# ------------------------
def build_rag(selected_lang: str):
    db = SessionLocal()
    rows = db.execute(select(prompt_answer)).fetchall()
    db.close()

    if not rows:
        return None

    docs = [Document(page_content=row.answer, metadata={"prompt": row.prompt}) for row in rows]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and latest user question, reformulate a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    system_prompt = (
        f"You are an assistant. Always respond in {selected_lang}.\n"
        f"If answer not found in the provided context, respond strictly with: '{DONT_KNOW[selected_lang]}'\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

# ------------------------
# Initialize sample prompts (if empty)
# ------------------------
def initialize_sample_prompts():
    db = SessionLocal()
    existing = db.execute(select(prompt_answer)).fetchall()
    if not existing:
        samples = [
            {"prompt": "What is AI?", "answer": "AI stands for Artificial Intelligence, the simulation of human intelligence by machines."},
            {"prompt": "What is Python?", "answer": "Python is a popular programming language used for web, data science, and AI applications."},
            {"prompt": "What is Streamlit?", "answer": "Streamlit is a Python framework for building interactive web apps easily."},
            {"prompt": "What is Machine Learning?", "answer": "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed."},
        ]
        db.execute(insert(prompt_answer), samples)
        db.commit()
    db.close()

initialize_sample_prompts()

# ------------------------
# Pydantic models
# ------------------------
class PromptIn(BaseModel):
    prompt: str
    answer: Optional[str] = None

class PromptOut(BaseModel):
    id: int
    prompt: str
    answer: str
    timestamp: datetime

class ChatIn(BaseModel):
    prompt: str
    conversation_id: Optional[int] = None
    language: Optional[str] = "English"

class ChatOut(BaseModel):
    answer: str
    conversation_id: int

class ConversationOut(BaseModel):
    id: int
    user_id: Optional[int]
    title: Optional[str]
    created_at: datetime

# ------------------------
# ROUTES (no auth)
# ------------------------

# --- Prompts: list ---
@app.get("/prompts", response_model=List[PromptOut])
def list_prompts():
    db = SessionLocal()
    rows = db.execute(select(prompt_answer)).fetchall()
    db.close()
    return [{"id": r.id, "prompt": r.prompt, "answer": r.answer, "timestamp": r.timestamp} for r in rows]

# --- Prompts: create ---
@app.post("/prompts", response_model=PromptOut)
def create_prompt(payload: PromptIn):
    text = payload.prompt.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Prompt required")
    answer_text = (payload.answer or "").strip()
    if not answer_text:
        raise HTTPException(status_code=400, detail="Answer required")
    db = SessionLocal()
    # duplicate check
    rows = db.execute(select(prompt_answer)).fetchall()
    for r in rows:
        if r.prompt.strip().lower() == text.lower():
            db.close()
            raise HTTPException(status_code=400, detail="Duplicate prompt exists")
    res = db.execute(insert(prompt_answer).values(prompt=text, answer=answer_text, timestamp=datetime.utcnow()))
    db.commit()
    new_id = res.inserted_primary_key[0]
    row = db.execute(select(prompt_answer).where(prompt_answer.c.id == new_id)).fetchone()
    db.close()
    return {"id": row.id, "prompt": row.prompt, "answer": row.answer, "timestamp": row.timestamp}

# --- Prompts: get by id ---
@app.get("/prompts/{prompt_id}", response_model=PromptOut)
def get_prompt(prompt_id: int):
    db = SessionLocal()
    row = db.execute(select(prompt_answer).where(prompt_answer.c.id == prompt_id)).fetchone()
    db.close()
    if not row:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"id": row.id, "prompt": row.prompt, "answer": row.answer, "timestamp": row.timestamp}

# --- Prompts: update ---
@app.put("/prompts/{prompt_id}", response_model=PromptOut)
def update_prompt(prompt_id: int, payload: PromptIn):
    db = SessionLocal()
    row = db.execute(select(prompt_answer).where(prompt_answer.c.id == prompt_id)).fetchone()
    if not row:
        db.close()
        raise HTTPException(status_code=404, detail="Prompt not found")
    new_prompt = payload.prompt.strip() if payload.prompt else row.prompt
    new_answer = (payload.answer or row.answer).strip()
    # collision check
    other_rows = db.execute(select(prompt_answer).where(prompt_answer.c.id != prompt_id)).fetchall()
    for orow in other_rows:
        if orow.prompt.strip().lower() == new_prompt.lower():
            db.close()
            raise HTTPException(status_code=400, detail="Another prompt with same text exists")
    db.execute(update(prompt_answer).where(prompt_answer.c.id == prompt_id)
               .values(prompt=new_prompt, answer=new_answer, timestamp=datetime.utcnow()))
    db.commit()
    updated = db.execute(select(prompt_answer).where(prompt_answer.c.id == prompt_id)).fetchone()
    db.close()
    return {"id": updated.id, "prompt": updated.prompt, "answer": updated.answer, "timestamp": updated.timestamp}

# --- Prompts: delete ---
@app.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: int):
    db = SessionLocal()
    row = db.execute(select(prompt_answer).where(prompt_answer.c.id == prompt_id)).fetchone()
    if not row:
        db.close()
        raise HTTPException(status_code=404, detail="Prompt not found")
    db.execute(delete(prompt_answer).where(prompt_answer.c.id == prompt_id))
    db.commit()
    db.close()
    return {"detail": "Deleted"}

# --- Conversations: list (no auth -> return all) ---
@app.get("/conversations", response_model=List[ConversationOut])
def list_conversations():
    db = SessionLocal()
    rows = db.execute(select(conversations)).fetchall()
    db.close()
    return [{"id": r.id, "user_id": r.user_id, "title": r.title, "created_at": r.created_at} for r in rows]

# --- Conversations: create ---
@app.post("/conversations", response_model=ConversationOut)
def create_conversation(title: Optional[str] = None, user_id: Optional[int] = None):
    db = SessionLocal()
    res = db.execute(insert(conversations).values(user_id=user_id or 0, title=title or "New Chat", created_at=datetime.utcnow()))
    db.commit()
    new_id = res.inserted_primary_key[0]
    row = db.execute(select(conversations).where(conversations.c.id == new_id)).fetchone()
    db.close()
    return {"id": row.id, "user_id": row.user_id, "title": row.title, "created_at": row.created_at}

# --- Conversations: delete ---
@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: int):
    db = SessionLocal()
    conv = db.execute(select(conversations).where(conversations.c.id == conv_id)).fetchone()
    if not conv:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.execute(delete(messages).where(messages.c.conversation_id == conv_id))
    db.execute(delete(conversations).where(conversations.c.id == conv_id))
    db.commit()
    db.close()
    return {"detail": "Conversation deleted"}

# --- Chat endpoint (no auth) ---
@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt required")

    language = (payload.language or "English").strip()
    if language not in ALLOWED_LANGS:
        raise HTTPException(status_code=400, detail=f"Language must be one of {list(ALLOWED_LANGS)}")

    db = SessionLocal()

    # Check exact existing prompt first (case-insensitive)
    rows = db.execute(select(prompt_answer)).fetchall()
    existing_row = None
    for r in rows:
        if r.prompt.strip().lower() == prompt.lower():
            existing_row = r
            break

    # Ensure (or create) conversation
    conv_id = payload.conversation_id
    if conv_id is None:
        res = db.execute(insert(conversations).values(user_id=0, title=prompt[:200], created_at=datetime.utcnow()))
        db.commit()
        conv_id = res.inserted_primary_key[0]
    else:
        conv = db.execute(select(conversations).where(conversations.c.id == conv_id)).fetchone()
        if not conv:
            db.close()
            raise HTTPException(status_code=404, detail="Conversation not found")

    if existing_row:
        answer = existing_row.answer
    else:
        rag = build_rag(language)
        if rag is None:
            answer = DONT_KNOW[language]
        else:
            try:
                result = rag.invoke({"input": prompt}, config={"configurable": {"session_id": str(conv_id)}})
                answer = result.get("answer", DONT_KNOW[language]) if isinstance(result, dict) else (getattr(result, "answer", None) or getattr(result, "content", None) or DONT_KNOW[language])
            except Exception:
                answer = DONT_KNOW[language]

        # Save unique prompt->answer
        db.execute(insert(prompt_answer).values(prompt=prompt, answer=answer, timestamp=datetime.utcnow()))
        db.commit()

    # Save messages
    db.execute(insert(messages).values(conversation_id=conv_id, role="user", content=prompt, timestamp=datetime.utcnow()))
    db.execute(insert(messages).values(conversation_id=conv_id, role="assistant", content=answer, timestamp=datetime.utcnow()))
    db.commit()
    db.close()

    return {"answer": answer, "conversation_id": conv_id}

if __name__ == "__main__":
    import uvicorn
    # Start server: uvicorn app:app --reload
    uvicorn.run("sample_check:app", host="0.0.0.0", port=8000, reload=True)
