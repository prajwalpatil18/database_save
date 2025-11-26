# app.py
"""
FastAPI backend version of your Streamlit RAG chat app.

Main features:
- SQLite DB with tables: users, conversations, messages, prompt_answer (same schema as original).
- HuggingFace embeddings cached; ChatGroq LLM.
- FAISS vectorstore built from prompt_answer rows.
- History-aware retriever + retrieval chain using LangChain (same pattern).
- JWT-based auth for users; simple admin/password endpoint.
- Endpoints:
    POST  /signup                -> create new user
    POST  /login                 -> login, returns JWT access_token
    POST  /admin/login           -> login as admin (password compare)
    GET   /prompts               -> list all prompt-answer pairs
    POST  /prompts               -> add prompt-answer (admin or protected)
    PUT   /prompts/{id}          -> edit prompt-answer
    DELETE /prompts/{id}         -> delete prompt-answer
    GET   /conversations         -> list user's conversations
    POST  /conversations        -> create new conversation (title optional)
    DELETE /conversations/{id}  -> delete conversation and messages
    POST  /chat                 -> main chat endpoint: { question, conversation_id? }
                                 returns assistant answer; stores messages & (if new) prompt_answer
- Use the same duplicate-checking logic (case-insensitive).
- NOTE: Adjust token secret / admin password / API keys in env for production.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Table, Column, Integer, String, ForeignKey, DateTime, Text, MetaData, select, insert, update, delete
)
from sqlalchemy.orm import sessionmaker
import bcrypt
import jwt  # PyJWT
from dotenv import load_dotenv

# LangChain / embeddings / retriever imports (from your original code)
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

load_dotenv()

# ------------------------
# Config & secrets
# ------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "replace-this-secret-in-prod")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # change in prod
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")  # HF token for embeddings
GROQ_API_KEY = os.getenv("API_KEY", "")  # Groq API Key

# ------------------------
# FastAPI app & DB setup
# ------------------------
app = FastAPI(title="RAG Chat Backend")

# SQLite engine & metadata
engine = create_engine("sqlite:///rag_chat_app.db", connect_args={"check_same_thread": False})
metadata = MetaData()

# Define tables (same as your Streamlit version)
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
SessionLocal = sessionmaker(bind=engine)

# ------------------------
# Embeddings & LLM initialization
# ------------------------
# Cache embeddings instance globally to avoid reloading each request
_embeddings_instance = None


def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    return _embeddings_instance


embeddings = get_embeddings()

# Initialize LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

# ------------------------
# Utility: JWT token helpers
# ------------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# Dependency to get current user from Authorization header
def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Expects Authorization: Bearer <token>
    Returns user dict {id, username} on success or raises 401.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    token = authorization.split(" ", 1)[1]
    payload = decode_access_token(token)
    user_id = payload.get("user_id")
    username = payload.get("username")
    if not user_id or not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return {"id": int(user_id), "username": username}


# ------------------------
# Initialize sample prompts if empty (same samples you used)
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
# In-memory session-store for message histories (used by RunnableWithMessageHistory)
# We replicate the st.session_state.store behavior: map conversation_id -> ChatMessageHistory
# ------------------------
SESSION_STORES: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Return a ChatMessageHistory for the given session_id, creating it if needed.
    RunnableWithMessageHistory expects a callable that returns a BaseChatMessageHistory-like object.
    """
    if session_id not in SESSION_STORES:
        SESSION_STORES[session_id] = ChatMessageHistory()
    return SESSION_STORES[session_id]


# ------------------------
# Build the retriever + RAG chain helper (rebuild when DB changes)
# ------------------------
# We'll keep the retriever and chain in global state and rebuild when prompts change.
RAG_STATE: Dict[str, Any] = {
    "vectorstore": None,
    "retriever": None,
    "conversational_rag_chain": None,
    # note: question_answer_chain, history_aware_retriever are rebuilt internally
}


def build_rag_chain():
    """
    Rebuild vectorstore, retriever and the RunnableWithMessageHistory chain from current DB prompt_answer rows.
    Call this whenever prompt_answer table is modified.
    """
    db = SessionLocal()
    rows = db.execute(select(prompt_answer)).fetchall()
    docs = [Document(page_content=row.answer, metadata={"prompt": row.prompt}) for row in rows]
    db.close()

    if not docs:
        RAG_STATE.update({"vectorstore": None, "retriever": None, "conversational_rag_chain": None})
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # Contextualization prompt (same as your code)
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", "Given a chat history and the latest user question, formulate a standalone question that can be understood without chat history. Do NOT answer, just reformulate if needed."),
         MessagesPlaceholder("chat_history"),
         ("human", "{input}")]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # System prompt for QA - will be instantiated at runtime to include the dont_know message & language
    # But we can create the chain skeleton now (actual system prompt will be created per request)
    # Use a placeholder substitution in request-time for system prompt text -> we'll pass ChatPromptTemplate later.

    # Build base stuff chain using a placeholder prompt (we'll recreate prompt template with language per call)
    dummy_system = "You are an assistant. {context}"
    qa_prompt = ChatPromptTemplate.from_messages([("system", dummy_system), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    RAG_STATE.update({
        "vectorstore": vectorstore,
        "retriever": retriever,
        "conversational_rag_chain": conversational_rag_chain
    })


# Build at startup
build_rag_chain()

# ------------------------
# Pydantic models for requests/responses
# ------------------------
class SignupIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PromptIn(BaseModel):
    prompt: str
    answer: str


class PromptOut(BaseModel):
    id: int
    prompt: str
    answer: str
    timestamp: datetime


class ConversationIn(BaseModel):
    title: Optional[str] = "New Chat"


class ConversationOut(BaseModel):
    id: int
    user_id: int
    title: str
    created_at: datetime


class ChatIn(BaseModel):
    question: str
    conversation_id: Optional[int] = None
    language: Optional[str] = "English"  # "English", "Hindi", "Marathi"


class ChatOut(BaseModel):
    answer: str
    conversation_id: Optional[int] = None


# ------------------------
# Auth endpoints
# ------------------------
@app.post("/signup", response_model=Dict[str, Any])
def signup(data: SignupIn):
    """
    Create new user with bcrypt hashed password.
    Returns a success message.
    """
    db = SessionLocal()
    existing = db.execute(select(users).where(users.c.username == data.username)).fetchone()
    if existing:
        db.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
    db.execute(insert(users).values(username=data.username, password_hash=hashed, created_at=datetime.utcnow()))
    db.commit()
    db.close()
    return {"message": "Signup successful"}


@app.post("/login", response_model=TokenOut)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2-form style login (username + password).
    Returns JWT token upon success.
    NOTE: To keep compatibility with your previous 'Prajwal' admin bypass,
    if username == "Prajwal" we bypass password check and return user_id 1 (if exists).
    """
    username = form_data.username
    password = form_data.password
    db = SessionLocal()
    user = db.execute(select(users).where(users.c.username == username)).fetchone()

    # admin bypass: username "Prajwal"
    if username == "Prajwal":
        # try to obtain user id if exists; if not, create it
        if not user:
            hashed = bcrypt.hashpw("password".encode(), bcrypt.gensalt()).decode()
            result = db.execute(insert(users).values(username="Prajwal", password_hash=hashed, created_at=datetime.utcnow()))
            db.commit()
            user_id = result.inserted_primary_key[0]
        else:
            user_id = user.id
        db.close()
        access_token = create_access_token({"user_id": user_id, "username": "Prajwal"})
        return TokenOut(access_token=access_token)

    if user and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        user_id = user.id
        db.close()
        access_token = create_access_token({"user_id": user_id, "username": username})
        return TokenOut(access_token=access_token)

    db.close()
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/admin/login")
def admin_login(password: Dict[str, str]):
    """
    Very simple admin login endpoint. Returns a short-lived admin token when password matches.
    (You may implement more robust admin handling later.)
    Request body: { "password": "admin123" }
    """
    provided = password.get("password")
    if provided is None:
        raise HTTPException(status_code=400, detail="Password required")
    if provided == ADMIN_PASSWORD:
        token = create_access_token({"user_id": -1, "username": "admin"}, expires_delta=timedelta(minutes=60))
        return {"access_token": token}
    else:
        raise HTTPException(status_code=401, detail="Incorrect admin password")


# ------------------------
# Prompt-answer management endpoints
# (GET/POST/PUT/DELETE). Protected by token (admin or normal users).
# When prompts change, we rebuild the RAG chain.
# ------------------------
@app.get("/prompts", response_model=List[PromptOut])
def list_prompts(current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    rows = db.execute(select(prompt_answer)).fetchall()
    db.close()
    return [{"id": r.id, "prompt": r.prompt, "answer": r.answer, "timestamp": r.timestamp} for r in rows]


@app.post("/prompts", response_model=PromptOut)
def add_prompt(p: PromptIn, current_user: dict = Depends(get_current_user)):
    """
    Add a new prompt->answer. Prevent duplicates (case-insensitive).
    After successful insert, rebuild the RAG chain so new data is available.
    """
    db = SessionLocal()
    rows = db.execute(select(prompt_answer)).fetchall()
    for r in rows:
        if r.prompt.strip().lower() == p.prompt.strip().lower():
            db.close()
            raise HTTPException(status_code=400, detail="A prompt with same text already exists (case-insensitive).")
    result = db.execute(insert(prompt_answer).values(prompt=p.prompt.strip(), answer=p.answer.strip(), timestamp=datetime.utcnow()))
    db.commit()
    new_id = result.inserted_primary_key[0]
    row = db.execute(select(prompt_answer).where(prompt_answer.c.id == new_id)).fetchone()
    db.close()

    # rebuild RAG chain to include this new doc
    build_rag_chain()

    return {"id": row.id, "prompt": row.prompt, "answer": row.answer, "timestamp": row.timestamp}


@app.put("/prompts/{prompt_id}", response_model=PromptOut)
def edit_prompt(prompt_id: int, p: PromptIn, current_user: dict = Depends(get_current_user)):
    """
    Update an existing prompt->answer while preventing case-insensitive collisions.
    Rebuild RAG chain after update.
    """
    db = SessionLocal()
    row = db.execute(select(prompt_answer).where(prompt_answer.c.id == prompt_id)).fetchone()
    if not row:
        db.close()
        raise HTTPException(status_code=404, detail="Prompt not found")

    # check collisions with other prompts
    other_rows = db.execute(select(prompt_answer).where(prompt_answer.c.id != prompt_id)).fetchall()
    for orow in other_rows:
        if orow.prompt.strip().lower() == p.prompt.strip().lower():
            db.close()
            raise HTTPException(status_code=400, detail="Another prompt with same text exists.")

    db.execute(update(prompt_answer).where(prompt_answer.c.id == prompt_id).values(prompt=p.prompt.strip(), answer=p.answer.strip(), timestamp=datetime.utcnow()))
    db.commit()
    updated = db.execute(select(prompt_answer).where(prompt_answer.c.id == prompt_id)).fetchone()
    db.close()

    # rebuild vectorstore
    build_rag_chain()

    return {"id": updated.id, "prompt": updated.prompt, "answer": updated.answer, "timestamp": updated.timestamp}


@app.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: int, current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    row = db.execute(select(prompt_answer).where(prompt_answer.c.id == prompt_id)).fetchone()
    if not row:
        db.close()
        raise HTTPException(status_code=404, detail="Prompt not found")
    db.execute(delete(prompt_answer).where(prompt_answer.c.id == prompt_id))
    db.commit()
    db.close()

    # Rebuild chain to remove the deleted doc
    build_rag_chain()
    return {"message": "Deleted successfully"}


# ------------------------
# Conversations & messages endpoints
# ------------------------
@app.post("/conversations", response_model=ConversationOut)
def create_conversation(data: ConversationIn, current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    result = db.execute(insert(conversations).values(user_id=current_user["id"], title=(data.title or "New Chat"), created_at=datetime.utcnow()))
    db.commit()
    conv_id = result.inserted_primary_key[0]
    row = db.execute(select(conversations).where(conversations.c.id == conv_id)).fetchone()
    db.close()
    return {"id": row.id, "user_id": row.user_id, "title": row.title, "created_at": row.created_at}


@app.get("/conversations", response_model=List[ConversationOut])
def list_conversations(current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    rows = db.execute(select(conversations).where(conversations.c.user_id == current_user["id"])).fetchall()
    db.close()
    return [{"id": r.id, "user_id": r.user_id, "title": r.title, "created_at": r.created_at} for r in rows]


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: int, current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    conv = db.execute(select(conversations).where(conversations.c.id == conv_id)).fetchone()
    if not conv:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.user_id != current_user["id"] and current_user["username"] != "admin":
        db.close()
        raise HTTPException(status_code=403, detail="Not allowed to delete this conversation")
    # delete messages and conversation
    db.execute(delete(messages).where(messages.c.conversation_id == conv_id))
    db.execute(delete(conversations).where(conversations.c.id == conv_id))
    db.commit()
    db.close()
    # Also clear session history for this conversation_id to avoid stale memory
    SESSION_STORES.pop(str(conv_id), None)
    return {"message": "Conversation deleted"}


@app.get("/conversations/{conv_id}/messages")
def get_conversation_messages(conv_id: int, current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    conv = db.execute(select(conversations).where(conversations.c.id == conv_id)).fetchone()
    if not conv:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.user_id != current_user["id"] and current_user["username"] != "admin":
        db.close()
        raise HTTPException(status_code=403, detail="Not allowed to view messages")
    msgs = db.execute(select(messages).where(messages.c.conversation_id == conv_id).order_by(messages.c.timestamp)).fetchall()
    db.close()
    return [{"id": m.id, "role": m.role, "content": m.content, "timestamp": m.timestamp} for m in msgs]


# ------------------------
# Main chat endpoint - this triggers the RAG flow
# ------------------------
@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn, current_user: dict = Depends(get_current_user)):
    """
    Accepts { question, conversation_id?, language }.
    Flow:
    - If conversation_id not provided, create a new conversation and set its title to the first question.
    - Check exact prompt in prompt_answer table (case-insensitive). If found, return stored answer (do NOT save duplicate).
    - Else, if no RAG chain available (no docs), return dont_know message in selected language.
    - Else, invoke the RunnableWithMessageHistory conversational chain, store messages in DB, and also insert new prompt_answer row (if not duplicate).
    - Return assistant answer and conversation_id.
    """
    question = payload.question.strip()
    language = payload.language or "English"
    if language not in ["English", "Hindi", "Marathi"]:
        language = "English"

    dont_know_responses = {
        "English": "I don't know about this.",
        "Hindi": "मुझे नहीं पता।",
        "Marathi": "मला माहित नाही."
    }

    db = SessionLocal()

    # If no conversation id provided -> create new conversation and use question as title
    conv_id = payload.conversation_id
    if not conv_id:
        res = db.execute(insert(conversations).values(user_id=current_user["id"], title=question[:200], created_at=datetime.utcnow()))
        db.commit()
        conv_id = res.inserted_primary_key[0]
    else:
        # validate conversation belongs to user (or admin)
        conv_row = db.execute(select(conversations).where(conversations.c.id == conv_id)).fetchone()
        if not conv_row:
            db.close()
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conv_row.user_id != current_user["id"] and current_user["username"] != "admin":
            db.close()
            raise HTTPException(status_code=403, detail="Not allowed to use this conversation")

        # If conversation title is "New Chat" or empty, update to first question
        if not conv_row.title or conv_row.title.strip().lower() == "new chat":
            db.execute(update(conversations).where(conversations.c.id == conv_id).values(title=question[:200]))
            db.commit()

    # Build docs again (fresh) so changes are picked up (like your Streamlit flow)
    rows = db.execute(select(prompt_answer)).fetchall()
    docs = [Document(page_content=row.answer, metadata={"prompt": row.prompt}) for row in rows]
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", "Given a chat history and the latest user question, formulate a standalone question that can be understood without chat history. Do NOT answer, just reformulate if needed."),
             MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Create a system prompt that includes the "dont know" message & language instruction
        system_prompt = (
            f"You are an assistant for question-answering tasks. "
            f"Use only the information available in the provided database to answer the user's question. "
            f"If the answer is not present, strictly respond with '{dont_know_responses[language]}' "
            f"If the context is in database then strictly respond in {language} language. Keep the answer concise (max 3 sentences). "
            f"The user may ask questions related to the provided context. You should respond intelligently, even if only a small portion of the context exists in the database, and strictly reply in {language}."
            f"Always strictly respond in {language} language.\n\n{{context}}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    else:
        conversational_rag_chain = None

    # Check existing prompt in DB (case-insensitive)
    existing_row = None
    for r in rows:
        if r.prompt.strip().lower() == question.lower():
            existing_row = r
            break

    # If exact prompt exists, just return stored answer (do not save again)
    if existing_row:
        answer = existing_row.answer
    else:
        # If no retriever/chain available, return dont_know in selected language
        if conversational_rag_chain is None:
            answer = dont_know_responses[language]
        else:
            # Append messages to session-bound ChatMessageHistory and invoke chain
            # Use conversation id as session id for history
            session_id = str(conv_id)
            # invoke the chain (synchronous)
            response = conversational_rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )
            answer = response["answer"]

        # Save new prompt->answer only if not duplicate and not empty
        normalized_prompt = question
        normalized_key = normalized_prompt.lower()
        if normalized_prompt:
            # double-check again before insert (race-safe-ish)
            rows_check = db.execute(select(prompt_answer)).fetchall()
            collision = False
            for r in rows_check:
                if r.prompt.strip().lower() == normalized_key:
                    collision = True
                    break
            if not collision:
                db.execute(insert(prompt_answer).values(prompt=normalized_prompt, answer=answer, timestamp=datetime.utcnow()))
                db.commit()
                # Optionally rebuild global RAG state to include new doc for future queries
                build_rag_chain()

    # Persist user & assistant messages in DB messages table
    db.execute(insert(messages).values(conversation_id=conv_id, role="user", content=question, timestamp=datetime.utcnow()))
    db.execute(insert(messages).values(conversation_id=conv_id, role="assistant", content=answer, timestamp=datetime.utcnow()))
    db.commit()
    db.close()

    return {"answer": answer, "conversation_id": conv_id}


# ------------------------
# Additional helpful endpoints (optional)
# ------------------------
@app.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    return {"id": current_user["id"], "username": current_user["username"]}


# ------------------------
# Run instructions (if run as main)
# ------------------------
if __name__ == "__main__":
    import uvicorn
    # Start server: uvicorn app:app --reload
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
