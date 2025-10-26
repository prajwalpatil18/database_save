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
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


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
# Initialize sample prompts
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
# Auth Helpers
# ------------------------
def signup_user(username, password):
    db = SessionLocal()
    existing = db.execute(select(users).where(users.c.username == username)).fetchone()
    if existing:
        return False, "Username already exists"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    db.execute(insert(users).values(username=username, password_hash=hashed))
    db.commit()
    db.close()
    return True, "Signup successful"

def login_user(username, password):
    db = SessionLocal()
    user = db.execute(select(users).where(users.c.username == username)).fetchone()
    db.close()
    if username == "Prajwal":  # Admin bypass
        return True, 1
    if user and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        return True, user.id
    return False, None

# ------------------------
# Admin Page
# ------------------------
def admin_page():
    st.title("🧠 Admin Dashboard – Manage Prompt & Answer Database")
    db = SessionLocal()
    st.sidebar.title("⚙️ Actions")
    action = st.sidebar.radio("Select Action:", ["View Database", "Add or Update Answer", "Logout"])

    if action == "View Database":
        st.subheader("📋 Prompt–Answer Table")
        data = db.execute(select(prompt_answer)).fetchall()
        if not data:
            st.warning("No records found in the database.")
        else:
            st.dataframe(
                [{"ID": row.id, "Prompt": row.prompt, "Answer": row.answer, "Timestamp": row.timestamp} for row in data]
            )

    elif action == "Add or Update Answer":
        st.subheader("✍️ Add / Update Prompt Answer")
        prompt_text = st.text_area("Enter Prompt")
        answer_text = st.text_area("Enter Answer")

        if st.button("Save to Database"):
            if not prompt_text.strip() or not answer_text.strip():
                st.warning("Both fields are required.")
            else:
                existing = db.execute(select(prompt_answer).where(prompt_answer.c.prompt == prompt_text)).fetchone()
                if existing:
                    db.execute(update(prompt_answer)
                               .where(prompt_answer.c.prompt == prompt_text)
                               .values(answer=answer_text, timestamp=datetime.utcnow()))
                    db.commit()
                    st.success("✅ Answer updated successfully!")
                else:
                    db.execute(insert(prompt_answer)
                               .values(prompt=prompt_text, answer=answer_text, timestamp=datetime.utcnow()))
                    db.commit()
                    st.success("✅ New prompt–answer pair added successfully!")

    elif action == "Logout":
        st.session_state.page = "main"
        st.success("Logged out successfully!")
        st.rerun()
    db.close()

# ------------------------
# Helper: Save Q&A
# ------------------------
def save_prompt_to_db(prompt, answer):
    db = SessionLocal()
    db.execute(insert(prompt_answer).values(prompt=prompt, answer=answer))
    db.commit()
    db.close()

def load_documents_from_db():
    db = SessionLocal()
    rows = db.execute(select(prompt_answer)).fetchall()
    db.close()
    docs = [Document(page_content=row.answer, metadata={"prompt": row.prompt}) for row in rows]
    return docs

# ------------------------
# PAGE STATE
# ------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"

# ------------------------
# MAIN PAGE
# ------------------------
if st.session_state.page == "main":
    st.title("🧠 Conversational RAG with Database")

    if st.button("Admin Login"):
        st.session_state.page = "admin_login"

    # Login / Signup Section
    if "user_id" not in st.session_state:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                success, user_id = login_user(username, password)
                if success:
                    st.session_state.user_id = user_id
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        with tab2:
            new_username = st.text_input("New Username", key="signup_user")
            new_password = st.text_input("New Password", type="password", key="signup_pass")
            if st.button("Sign Up"):
                success, msg = signup_user(new_username, new_password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    else:
        st.write("Welcome! You’re logged in. (Chat UI code continues here...)")

# ------------------------
# ADMIN LOGIN PAGE
# ------------------------
elif st.session_state.page == "admin_login":
    st.title("🔑 Admin Login")
    password = st.text_input("Enter Admin Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit"):
            if password == "admin123":
                st.session_state.page = "admin_dashboard"
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Incorrect password ❌")
    with col2:
        if st.button("Back"):
            st.session_state.page = "main"

# ------------------------
# ADMIN DASHBOARD PAGE
# ------------------------
elif st.session_state.page == "admin_dashboard":
    admin_page()
