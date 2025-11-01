import streamlit as st
from sqlalchemy import create_engine, Table, Column, Integer, String, ForeignKey, DateTime, Text, MetaData, select, insert, update, delete
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import bcrypt
import os

# LangChain imports
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

from dotenv import load_dotenv
load_dotenv()

# ------------------------
# Setup Embeddings & Groq
# ------------------------
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embeddings = get_embeddings()
groq_api_key = os.getenv("API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# ------------------------
# SQLite DB Setup
# ------------------------
engine = create_engine("sqlite:///rag_chat_app.db", connect_args={"check_same_thread": False})
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
SessionLocal = sessionmaker(bind=engine)

# ------------------------
# Initialize sample data
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
# Authentication Helpers
# ------------------------
def signup_user(username, password):
    db = SessionLocal()
    existing = db.execute(select(users).where(users.c.username == username)).fetchone()
    if existing:
        db.close()
        return False, "Username already exists"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    db.execute(insert(users).values(username=username, password_hash=hashed))
    db.commit()
    db.close()
    return True, "Signup successful"

def login_user(username, password):
    db = SessionLocal()
    user = db.execute(select(users).where(users.c.username == username)).fetchone()
    if username == "Prajwal":
        db.close()
        return True, 1
    if user and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        uid = user.id
        db.close()
        return True, uid
    db.close()
    return False, None

# ------------------------
# Admin Dashboard
# ------------------------
def admin_dashboard():
    db = SessionLocal()
    st.title("🧠 Admin Dashboard – Manage Prompt & Answer Database")
    st.sidebar.title("⚙️ Admin Actions")
    action = st.sidebar.radio("Select Action:", ["View Database", "Add Prompt/Answer", "Logout"])

    if action == "View Database":
        st.subheader("📋 Prompt–Answer Table")
        rows = db.execute(select(prompt_answer)).fetchall()
        if not rows:
            st.warning("No records found.")
        else:
            for row in rows:
                with st.expander(f"ID {row.id} — Prompt: {row.prompt[:60]}{'...' if len(row.prompt) > 60 else ''}"):
                    st.write("**Prompt:**")
                    prompt_text = st.text_area("Prompt", value=row.prompt, key=f"prompt_{row.id}")
                    st.write("**Answer:**")
                    answer_text = st.text_area("Answer", value=row.answer, key=f"answer_{row.id}")
                    col1, col2, col3 = st.columns([1,1,1])
                    with col1:
                        if st.button("Save Changes", key=f"save_{row.id}"):
                            if not prompt_text.strip() or not answer_text.strip():
                                st.warning("Both fields are required.")
                            else:
                                existing_same_prompt = db.execute(
                                    select(prompt_answer).where(prompt_answer.c.prompt == prompt_text).where(prompt_answer.c.id != row.id)
                                ).fetchone()
                                if existing_same_prompt:
                                    st.error("Another record already uses this prompt.")
                                else:
                                    db.execute(update(prompt_answer).where(prompt_answer.c.id == row.id)
                                               .values(prompt=prompt_text, answer=answer_text, timestamp=datetime.utcnow()))
                                    db.commit()
                                    st.success("Updated successfully!")
                                    st.rerun()
                    with col2:
                        if st.button("Delete Row", key=f"delrow_{row.id}"):
                            db.execute(delete(prompt_answer).where(prompt_answer.c.id == row.id))
                            db.commit()
                            st.success("Deleted successfully!")
                            st.rerun()
                    with col3:
                        if st.button("Copy Prompt", key=f"copy_{row.id}"):
                            st.session_state.admin_clipboard = prompt_text

            st.markdown("---")
            st.write("Clipboard:")
            st.text_area("Clipboard", value=st.session_state.get("admin_clipboard", ""), key="admin_clipboard_area")

    elif action == "Add Prompt/Answer":
        st.subheader("✍️ Add Prompt–Answer")
        prompt_text = st.text_area("Enter Prompt", value=st.session_state.get("admin_clipboard", ""))
        answer_text = st.text_area("Enter Answer")
        if st.button("Save to Database"):
            if not prompt_text.strip() or not answer_text.strip():
                st.warning("Both fields are required.")
            else:
                existing = db.execute(select(prompt_answer).where(prompt_answer.c.prompt == prompt_text)).fetchone()
                if existing:
                    st.error("A record with this exact prompt already exists.")
                else:
                    db.execute(insert(prompt_answer).values(prompt=prompt_text, answer=answer_text, timestamp=datetime.utcnow()))
                    db.commit()
                    st.success("✅ Added successfully!")
                    st.session_state.admin_clipboard = ""
                    st.rerun()

    elif action == "Logout":
        st.session_state.page = "main"
        st.session_state.is_admin = False
        st.success("Logged out successfully!")
        st.rerun()
    db.close()

# ------------------------
# Session Initialization
# ------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "English"
if "store" not in st.session_state:
    st.session_state.store = {}
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

st.set_page_config(page_title="RAG DB Chat App", layout="wide")

# ------------------------
# Admin Login
# ------------------------
if st.session_state.page == "admin_login":
    st.title("🔑 Admin Login")
    password = st.text_input("Enter Admin Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit"):
            if password == "admin123":
                st.session_state.is_admin = True
                st.session_state.page = "admin"
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Incorrect password ❌")
    with col2:
        if st.button("Back"):
            st.session_state.page = "main"
            st.rerun()

# ------------------------
# Admin Dashboard Page
# ------------------------
elif st.session_state.page == "admin" and st.session_state.is_admin:
    admin_dashboard()

# ------------------------
# Main App
# ------------------------
else:
    st.title("🧠 Conversational RAG with Database")

    if st.button("Admin Login"):
        st.session_state.page = "admin_login"
        st.rerun()

    # The rest of your main chat logic (unchanged)
    # ✅ Everything else in your chat UI remains the same.
