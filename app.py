# main.py
import os
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

import bcrypt
import uvicorn
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import (Column, DateTime, ForeignKey, Integer, MetaData, String,
                        Table, Text, create_engine, delete, insert, select,
                        update)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# LangChain / embeddings / vectorstore imports
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

load_dotenv()

# ------------------------
# Config & Environment
# ------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_chat_app.db")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("API_KEY")  # your groq key
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
DEFAULT_LANGUAGE = "English"

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# ------------------------
# Database Setup (Tables mirror your Streamlit app)
# ------------------------
engine: Engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
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
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("title", String, default="New Chat"),
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
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# ------------------------
# Embeddings & LLM (cached)
# ------------------------
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

@lru_cache(maxsize=1)
def get_llm():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ API key not set (API_KEY env var)")
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

# ------------------------
# Pydantic Schemas
# ------------------------
class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class PromptAnswerCreate(BaseModel):
    prompt: str
    answer: str

class PromptAnswerOut(BaseModel):
    id: int
    prompt: str
    answer: str
    timestamp: datetime

class ConversationCreate(BaseModel):
    user_id: int
    title: Optional[str] = "New Chat"

class ChatRequest(BaseModel):
    user_id: int
    conversation_id: Optional[int] = None
    prompt: str
    language: Optional[str] = DEFAULT_LANGUAGE

class ChatResponse(BaseModel):
    answer: str
    conversation_id: int

# ------------------------
# Utility Functions
# ------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

# Initialize embeddings & llm (will raise if missing)
embeddings = get_embeddings()
llm = get_llm()

# ------------------------
# Helper: Build retriever from DB (current prompt_answer rows)
# ------------------------
def build_retriever_from_db(db_session) -> Optional[FAISS]:
    rows = db_session.execute(select(prompt_answer)).fetchall()
    if not rows:
        return None
    docs = [Document(page_content=row.answer, metadata={"prompt": row.prompt}) for row in rows]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

# ------------------------
# History-aware retriever & chain builder
# ------------------------
def build_rag_chain(retriever):
    if retriever is None:
        return None

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question that can be understood "
        "without chat history. Do NOT answer, just reformulate if needed."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    dont_know = {
        "English": "I don't know about this.",
        "Hindi": "मुझे नहीं पता।",
        "Marathi": "मला माहित नाही."
    }

    system_prompt_template = (
        "You are an assistant for question-answering tasks. "
        "Use only the information available in the provided database to answer the user's question. "
        "If the answer is not present, strictly respond with '{dont_know}' "
        "Keep the answer concise (max 3 sentences). Always strictly respond in {language}.\n\n{{context}}"
    )

    def create_chain_for_language(language):
        dont_know_text = dont_know.get(language, dont_know["English"])
        system_prompt = system_prompt_template.format(dont_know=dont_know_text, language=language)
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain

    return create_chain_for_language

# ------------------------
# FastAPI app & endpoints
# ------------------------
app = FastAPI(title="RAG Chat Backend")

@app.on_event("startup")
def startup_tasks():
    # initialize sample prompts if empty (same as Streamlit)
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

# ------------------------
# Auth endpoints
# ------------------------
@app.post("/signup", status_code=201)
def signup(payload: SignupRequest, db=Depends(get_db)):
    username = payload.username.strip()
    if not username or not payload.password:
        raise HTTPException(status_code=400, detail="username and password required")
    existing = db.execute(select(users).where(users.c.username == username)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed = hash_password(payload.password)
    db.execute(insert(users).values(username=username, password_hash=hashed, created_at=datetime.utcnow()))
    db.commit()
    return {"ok": True, "message": "Signup successful"}

@app.post("/login")
def login(payload: LoginRequest, db=Depends(get_db)):
    username = payload.username.strip()
    user = db.execute(select(users).where(users.c.username == username)).fetchone()
    # admin bypass for "Prajwal" as original code
    if username == "Prajwal":
        return {"ok": True, "user_id": 1}
    if user and verify_password(payload.password, user.password_hash):
        return {"ok": True, "user_id": user.id}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# ------------------------
# Admin endpoints (basic password check via header/query param)
# ------------------------
def check_admin(passwd: str = Query(..., alias="admin_password")):
    if passwd != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Forbidden: wrong admin password")

@app.get("/admin/prompts", response_model=List[PromptAnswerOut])
def list_prompts(admin_password: str = Depends(check_admin), db=Depends(get_db)):
    rows = db.execute(select(prompt_answer)).fetchall()
    return [PromptAnswerOut(id=r.id, prompt=r.prompt, answer=r.answer, timestamp=r.timestamp) for r in rows]

@app.post("/admin/prompts", response_model=PromptAnswerOut, status_code=201)
def create_prompt(item: PromptAnswerCreate, admin_password: str = Depends(check_admin), db=Depends(get_db)):
    prompt_text = item.prompt.strip()
    answer_text = item.answer.strip()
    if not prompt_text or not answer_text:
        raise HTTPException(status_code=400, detail="Both prompt and answer required")
    # case-insensitive duplicate check
    rows = db.execute(select(prompt_answer)).fetchall()
    for r in rows:
        if r.prompt.strip().lower() == prompt_text.lower():
            raise HTTPException(status_code=400, detail="Prompt already exists")
    res = db.execute(insert(prompt_answer).values(prompt=prompt_text, answer=answer_text, timestamp=datetime.utcnow()))
    db.commit()
    inserted_id = res.inserted_primary_key[0]
    row = db.execute(select(prompt_answer).where(prompt_answer.c.id == inserted_id)).fetchone()
    return PromptAnswerOut(id=row.id, prompt=row.prompt, answer=row.answer, timestamp=row.timestamp)

@app.put("/admin/prompts/{pid}", response_model=PromptAnswerOut)
def update_prompt(pid: int, item: PromptAnswerCreate, admin_password: str = Depends(check_admin), db=Depends(get_db)):
    row = db.execute(select(prompt_answer).where(prompt_answer.c.id == pid)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    # check collision
    rows = db.execute(select(prompt_answer).where(prompt_answer.c.id != pid)).fetchall()
    for r in rows:
        if r.prompt.strip().lower() == item.prompt.strip().lower():
            raise HTTPException(status_code=400, detail="Another prompt with same text exists")
    db.execute(update(prompt_answer).where(prompt_answer.c.id == pid).values(prompt=item.prompt.strip(), answer=item.answer.strip(), timestamp=datetime.utcnow()))
    db.commit()
    updated = db.execute(select(prompt_answer).where(prompt_answer.c.id == pid)).fetchone()
    return PromptAnswerOut(id=updated.id, prompt=updated.prompt, answer=updated.answer, timestamp=updated.timestamp)

@app.delete("/admin/prompts/{pid}", status_code=204)
def delete_prompt(pid: int, admin_password: str = Depends(check_admin), db=Depends(get_db)):
    db.execute(delete(prompt_answer).where(prompt_answer.c.id == pid))
    db.commit()
    return {}

# ------------------------
# Conversations & messages endpoints
# ------------------------
@app.post("/conversations", status_code=201)
def create_conversation(payload: ConversationCreate, db=Depends(get_db)):
    res = db.execute(insert(conversations).values(user_id=payload.user_id, title=payload.title, created_at=datetime.utcnow()))
    db.commit()
    conv_id = res.inserted_primary_key[0]
    return {"conversation_id": conv_id}

@app.get("/conversations/{user_id}")
def list_conversations(user_id: int, db=Depends(get_db)):
    convs = db.execute(select(conversations).where(conversations.c.user_id == user_id)).fetchall()
    out = [{"id": c.id, "title": c.title, "created_at": c.created_at} for c in convs]
    return out

@app.delete("/conversations/{conv_id}", status_code=204)
def delete_conversation(conv_id: int, db=Depends(get_db)):
    db.execute(delete(messages).where(messages.c.conversation_id == conv_id))
    db.execute(delete(conversations).where(conversations.c.id == conv_id))
    db.commit()
    return {}

# ------------------------
# Chat endpoint: core RAG logic
# ------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db=Depends(get_db)):
    user_id = payload.user_id
    prompt = payload.prompt.strip()
    language = payload.language or DEFAULT_LANGUAGE

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt required")

    # If no conversation id, create one and set title to first question
    conv_id = payload.conversation_id
    if not conv_id:
        res = db.execute(insert(conversations).values(user_id=user_id, title=prompt[:200], created_at=datetime.utcnow()))
        db.commit()
        conv_id = res.inserted_primary_key[0]

    # Save user message
    db.execute(insert(messages).values(conversation_id=conv_id, role="user", content=prompt, timestamp=datetime.utcnow()))
    db.commit()

    # Check if exact prompt already exists in DB (case-insensitive)
    rows_all = db.execute(select(prompt_answer)).fetchall()
    existing_row = None
    for r in rows_all:
        if r.prompt.strip().lower() == prompt.lower():
            existing_row = r
            break

    # If exact match exists, return stored answer
    if existing_row:
        answer = existing_row.answer
    else:
        # Build retriever and chain
        retriever = build_retriever_from_db(db)
        if retriever is None:
            # no knowledge base -> don't know response
            dont_know = {
                "English": "I don't know about this.",
                "Hindi": "मुझे नहीं पता।",
                "Marathi": "मला माहित नाही."
            }
            answer = dont_know.get(language, dont_know["English"])
        else:
            # Build history-aware chain generator for language
            chain_creator = build_rag_chain(retriever)
            rag_chain = chain_creator(language)
            # session history object
            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                # Use a simple in-memory ChatMessageHistory - ephemeral per process
                # For production, persist history if needed.
                return ChatMessageHistory()
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            # invoke chain (synchronous)
            response = conversational_rag_chain.invoke({"input": prompt}, config={"configurable": {"session_id": str(conv_id)}})
            answer = response.get("answer", "")

        # Insert new prompt->answer into DB for future exact matches (same behavior as Streamlit)
        if prompt and not existing_row:
            # double-check collision before insert
            rows_check = db.execute(select(prompt_answer)).fetchall()
            collision = False
            for r in rows_check:
                if r.prompt.strip().lower() == prompt.lower():
                    collision = True
                    break
            if not collision:
                db.execute(insert(prompt_answer).values(prompt=prompt, answer=answer, timestamp=datetime.utcnow()))
                db.commit()

    # Save assistant message
    db.execute(insert(messages).values(conversation_id=conv_id, role="assistant", content=answer, timestamp=datetime.utcnow()))
    db.commit()

    return ChatResponse(answer=answer, conversation_id=conv_id)

# ------------------------
# Simple health check
# ------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
# ------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
