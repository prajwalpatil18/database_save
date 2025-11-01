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

@st.cache_resource  # cache resource globally using Streamlit’s cache
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

# User Table
users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String, unique=True, nullable=False),
    Column("password_hash", String, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow)
)

# Conversations Table
conversations = Table(
    "conversations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("title", String, default="New Chat"),
    Column("created_at", DateTime, default=datetime.utcnow)
)

# Messages Table
messages = Table(
    "messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("conversation_id", Integer, ForeignKey("conversations.id")),
    Column("role", String),
    Column("content", Text),
    Column("timestamp", DateTime, default=datetime.utcnow)
)

# Table for Prompts and Answers
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
    # Admin bypass - NOTE: keep as you intended; can change later
    if username == "Prajwal":
        db.close()
        return True, 1  # Admin bypass (user id 1)
    if user and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        uid = user.id
        db.close()
        return True, uid
    db.close()
    return False, None

# ------------------------
# Admin Dashboard (embedded) - extended with edit/delete
# ------------------------
def admin_dashboard():
    db = SessionLocal()
    st.title("🧠 Admin Dashboard – Manage Prompt & Answer Database")
    st.sidebar.title("⚙️ Admin Actions")
    action = st.sidebar.radio("Select Action:", ["View Database", "Add Prompt/Answer", "Logout"])

    # View Data with edit/delete
    if action == "View Database":
        st.subheader("📋 Prompt–Answer Table")
        rows = db.execute(select(prompt_answer)).fetchall()
        if not rows:
            st.warning("No records found.")
        else:
            # Show a compact table and allow editing row-by-row using expanders
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
                                # If prompt changed to a value that already exists under a different id, prevent duplicate keys
                                existing_same_prompt = db.execute(
                                    select(prompt_answer).where(prompt_answer.c.prompt == prompt_text).where(prompt_answer.c.id != row.id)
                                ).fetchone()
                                if existing_same_prompt:
                                    st.error("Another record already uses this prompt. Choose a different prompt (duplicates are not allowed).")
                                else:
                                    db.execute(update(prompt_answer).where(prompt_answer.c.id == row.id)
                                               .values(prompt=prompt_text, answer=answer_text, timestamp=datetime.utcnow()))
                                    db.commit()
                                    st.success("Updated successfully!")
                                    st.experimental_rerun()
                    with col2:
                        if st.button("Delete Row", key=f"delrow_{row.id}"):
                            # delete row and any references? Only prompt_answer table, fine
                            db.execute(delete(prompt_answer).where(prompt_answer.c.id == row.id))
                            db.commit()
                            st.success("Deleted successfully!")
                            st.experimental_rerun()
                    with col3:
                        if st.button("Copy Prompt", key=f"copy_{row.id}"):
                            st.write("Prompt copied to clipboard area below (use it in Add Prompt/Answer).")
                            st.session_state.admin_clipboard = prompt_text
            # clipboard area
            st.markdown("---")
            st.write("Admin clipboard (paste into Add Prompt/Answer if needed):")
            st.text_area("Clipboard", value=st.session_state.get("admin_clipboard", ""), key="admin_clipboard_area")

    # Add Prompt/Answer
    elif action == "Add Prompt/Answer":
        st.subheader("✍️ Add Prompt–Answer")
        # allow prefill from clipboard
        prompt_text = st.text_area("Enter Prompt", value=st.session_state.get("admin_clipboard", ""))
        answer_text = st.text_area("Enter Answer")
        if st.button("Save to Database"):
            if not prompt_text.strip() or not answer_text.strip():
                st.warning("Both fields are required.")
            else:
                # Prevent duplicates: exact match on prompt
                existing = db.execute(select(prompt_answer).where(prompt_answer.c.prompt == prompt_text)).fetchone()
                if existing:
                    st.error("A record with this exact prompt already exists. Use View Database to edit it.")
                else:
                    db.execute(insert(prompt_answer).values(prompt=prompt_text, answer=answer_text, timestamp=datetime.utcnow()))
                    db.commit()
                    st.success("✅ Added successfully!")
                    st.session_state.admin_clipboard = ""  # clear clipboard after add
                    st.experimental_rerun()

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

# for keeping UI consistent
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

    # ------------------------
    # Login / Signup
    # ------------------------
    if st.session_state.user_id is None:
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

    # ------------------------
    # Chat UI (after login)
    # ------------------------
    else:
        db = SessionLocal()
        st.sidebar.subheader("🌐 Choose Response Language")
        st.session_state.language = st.sidebar.radio(
            "Select a language for responses:",
            ["English", "Hindi", "Marathi"],
            index=0 if st.session_state.language == "English" else (1 if st.session_state.language == "Hindi" else 2)
        )

        # Load docs
        rows = db.execute(select(prompt_answer)).fetchall()
        docs = [Document(page_content=row.answer, metadata={"prompt": row.prompt}) for row in rows]

        if not docs:
            st.warning("No data found in the knowledge DB.")
            retriever = None
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever()

        # Sidebar: Conversations list (like ChatGPT)
        st.sidebar.title("💬 Conversations")
        convs = db.execute(select(conversations).where(conversations.c.user_id == st.session_state.user_id).order_by(conversations.c.created_at.desc())).fetchall()

        # show each conversation with its title (first question) and allow deleting
        for conv in convs:
            col1, col2 = st.sidebar.columns([4,1])
            with col1:
                if st.button(conv.title, key=f"conv_{conv.id}"):
                    st.session_state.active_conversation = conv.id
                    # load conversation messages
                    msgs = db.execute(select(messages).where(messages.c.conversation_id == conv.id).order_by(messages.c.timestamp)).fetchall()
                    st.session_state.messages = [{"role": m.role, "content": m.content} for m in msgs]
                    st.experimental_rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv.id}"):
                    db.execute(delete(messages).where(messages.c.conversation_id == conv.id))
                    db.execute(delete(conversations).where(conversations.c.id == conv.id))
                    db.commit()
                    if st.session_state.active_conversation == conv.id:
                        st.session_state.active_conversation = None
                        st.session_state.messages = []
                    st.success(f"Conversation '{conv.title}' deleted.")
                    st.experimental_rerun()

        if st.sidebar.button("➕ New Chat"):
            # create a placeholder conversation; title will be set to first prompt when user enters it
            result = db.execute(insert(conversations).values(user_id=st.session_state.user_id, title="New Chat", created_at=datetime.utcnow()))
            db.commit()
            new_conv_id = result.inserted_primary_key[0]
            st.session_state.active_conversation = new_conv_id
            st.session_state.messages = []
            st.experimental_rerun()

        if st.sidebar.button("🚪 Logout"):
            st.session_state.user_id = None
            st.session_state.active_conversation = None
            st.session_state.messages = []
            st.experimental_rerun()

        # Build RAG components if retriever exists
        if retriever:
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question, "
                "formulate a standalone question that can be understood "
                "without chat history. Do NOT answer, just reformulate if needed."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
            )
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            selected_lang = st.session_state.language
            dont_know_responses = {
                "English": "I don't know about this.",
                "Hindi": "मुझे नहीं पता।",
                "Marathi": "मला याबद्दल माहिती नाही."
            }

            system_prompt = (
                f"You are an assistant for question-answering tasks. "
                f"Use only the information available in the provided database to answer the user's question. "
                f"If the answer is not present, strictly respond with '{dont_know_responses[selected_lang]}' "
                f"If the context is in database then strictly respond in {selected_lang} language. Keep the answer concise (max 3 sentences). "
                f"The user may ask questions related to the provided context. You should respond intelligently, even if only a small portion of the context exists in the database, and strictly reply in {selected_lang}."
                f"Always strictly respond in {selected_lang} language.\n\n{{context}}"
            )

            qa_prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
            )
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
        else:
            conversational_rag_chain = None

        # Render existing messages in chat window
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        lang_placeholder = (
            "Enter your question..." if st.session_state.language == "English"
            else ("अपना प्रश्न दर्ज करें..." if st.session_state.language == "Hindi" else "आपला प्रश्न टाका...")
        )

        if prompt := st.chat_input(lang_placeholder):
            # write user message to UI & session
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # if there's no active conversation, create one now and set its title to the first prompt
            if not st.session_state.active_conversation:
                result = db.execute(insert(conversations).values(user_id=st.session_state.user_id, title=prompt[:200], created_at=datetime.utcnow()))
                db.commit()
                st.session_state.active_conversation = result.inserted_primary_key[0]
            else:
                # If active conversation has default "New Chat" or "New Chat" title, and has no messages, set it to this prompt
                conv_row = db.execute(select(conversations).where(conversations.c.id == st.session_state.active_conversation)).fetchone()
                if conv_row and (not conv_row.title or conv_row.title == "New Chat"):
                    db.execute(update(conversations).where(conversations.c.id == st.session_state.active_conversation).values(title=prompt[:200]))
                    db.commit()

            # First check for existing prompt (exact match). If exists, do NOT save duplicate later
            existing = db.execute(select(prompt_answer).where(prompt_answer.c.prompt == prompt)).fetchone()

            # Rebuild docs & retriever for up-to-date DB (in case admin changed it during session)
            rows = db.execute(select(prompt_answer)).fetchall()
            docs = [Document(page_content=row.answer, metadata={"prompt": row.prompt}) for row in rows]
            if not docs:
                st.warning("No data found in the knowledge DB.")
            else:
                embeddings = get_embeddings()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
                vectorstore = FAISS.from_documents(splits, embeddings)
                retriever = vectorstore.as_retriever()

                # rebuild conversational rag chain because retriever changed
                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
                )
                history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
                qa_prompt = ChatPromptTemplate.from_messages(
                    [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
                )
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain, get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )

            # If exact existing prompt found, use its answer and DO NOT insert a duplicate
            if existing:
                answer = existing.answer
            else:
                if conversational_rag_chain is None:
                    answer = dont_know_responses[st.session_state.language]
                else:
                    with st.spinner("Thinking..."):
                        session_history = get_session_history(str(st.session_state.active_conversation))
                        response = conversational_rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": str(st.session_state.active_conversation)}}
                        )
                        answer = response["answer"]

                # Save new prompt-answer pair only if it doesn't already exist (double-check) and answer is not empty
                existing_check = db.execute(select(prompt_answer).where(prompt_answer.c.prompt == prompt)).fetchone()
                if not existing_check and answer and answer.strip() and answer not in [dont_know_responses["English"], dont_know_responses["Hindi"], dont_know_responses["Marathi"]]:
                    # Insert new QA pair
                    db.execute(insert(prompt_answer).values(prompt=prompt, answer=answer, timestamp=datetime.utcnow()))
                    db.commit()

            # Show assistant message
            st.chat_message("assistant").write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Persist messages into messages table for the active conversation
            if st.session_state.active_conversation:
                db.execute(insert(messages).values(
                    conversation_id=st.session_state.active_conversation,
                    role="user",
                    content=prompt,
                    timestamp=datetime.utcnow()
                ))
                db.execute(insert(messages).values(
                    conversation_id=st.session_state.active_conversation,
                    role="assistant",
                    content=answer,
                    timestamp=datetime.utcnow()
                ))
                db.commit()

            # after message, refresh sidebar conversation list so title changes appear
            st.experimental_rerun()

        db.close()
