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

@st.cache_resource  # cache embeddings
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

# ✅ Table for Prompts and Answers
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
    # admin bypass for user "Prajwal" (keeps existing behavior)
    if username == "Prajwal":
        db.close()
        return True, 1
    if user and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        user_id = user.id
        db.close()
        return True, user_id
    db.close()
    return False, None

# ------------------------
# Admin Dashboard (embedded) - Enhanced with edit/delete
# ------------------------
def admin_dashboard():
    db = SessionLocal()
    st.title("🧠 Admin Dashboard – Manage Prompt & Answer Database")
    st.sidebar.title("⚙️ Admin Actions")
    action = st.sidebar.radio("Select Action:", ["View Database", "Add New Prompt", "Edit / Delete Prompt", "Logout"])

    # View Data
    if action == "View Database":
        st.subheader("📋 Prompt–Answer Table")
        data = db.execute(select(prompt_answer)).fetchall()
        if not data:
            st.warning("No records found.")
        else:
            st.dataframe(
                [{"ID": row.id, "Prompt": row.prompt, "Answer": row.answer, "Timestamp": row.timestamp} for row in data]
            )

    # Add New
    elif action == "Add New Prompt":
        st.subheader("✍️ Add Prompt–Answer")
        prompt_text = st.text_area("Enter Prompt")
        answer_text = st.text_area("Enter Answer")
        if st.button("Add to Database"):
            if not prompt_text.strip() or not answer_text.strip():
                st.warning("Both fields are required.")
            else:
                # check duplicate (case-insensitive)
                existing_rows = db.execute(select(prompt_answer)).fetchall()
                found = False
                for r in existing_rows:
                    if r.prompt.strip().lower() == prompt_text.strip().lower():
                        found = True
                        break
                if found:
                    st.warning("A prompt with same text already exists (case-insensitive). Use Edit to modify it.")
                else:
                    db.execute(insert(prompt_answer).values(prompt=prompt_text.strip(), answer=answer_text.strip(), timestamp=datetime.utcnow()))
                    db.commit()
                    st.success("✅ Added successfully!")
                    st.rerun()

    # Edit / Delete
    elif action == "Edit / Delete Prompt":
        st.subheader("🛠️ Edit or Delete Existing Prompt–Answer")
        rows = db.execute(select(prompt_answer)).fetchall()
        if not rows:
            st.info("No prompts to edit.")
        else:
            options = [f"{r.id} - {r.prompt[:80].replace(chr(10),' ')}" for r in rows]
            choice = st.selectbox("Select a prompt to edit", options)
            if choice:
                chosen_id = int(choice.split(" - ")[0])
                row = db.execute(select(prompt_answer).where(prompt_answer.c.id == chosen_id)).fetchone()
                if row:
                    new_prompt = st.text_area("Prompt", value=row.prompt, key=f"edit_prompt_{row.id}")
                    new_answer = st.text_area("Answer", value=row.answer, key=f"edit_answer_{row.id}")
                    col1, col2, col3 = st.columns([1,1,1])
                    with col1:
                        if st.button("Save Changes", key=f"save_{row.id}"):
                            if not new_prompt.strip() or not new_answer.strip():
                                st.warning("Both fields are required.")
                            else:
                                # prevent collision with other prompts (case-insensitive)
                                other_rows = db.execute(select(prompt_answer).where(prompt_answer.c.id != row.id)).fetchall()
                                collision = False
                                for orow in other_rows:
                                    if orow.prompt.strip().lower() == new_prompt.strip().lower():
                                        collision = True
                                        break
                                if collision:
                                    st.error("Another prompt with same text exists. Change text to avoid duplicates.")
                                else:
                                    db.execute(update(prompt_answer)
                                               .where(prompt_answer.c.id == row.id)
                                               .values(prompt=new_prompt.strip(), answer=new_answer.strip(), timestamp=datetime.utcnow()))
                                    db.commit()
                                    st.success("Updated successfully.")
                                    st.rerun()
                    with col2:
                        if st.button("Delete Prompt", key=f"del_prompt_{row.id}"):
                            db.execute(delete(prompt_answer).where(prompt_answer.c.id == row.id))
                            db.commit()
                            st.success("Deleted successfully.")
                            st.rerun()
                    with col3:
                        if st.button("Cancel"):
                            st.info("No changes made.")
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
        # Add Marathi option
        st.session_state.language = st.sidebar.radio(
            "Select a language for responses:",
            ["English", "Hindi", "Marathi"],
            index=0 if st.session_state.language == "English" else (1 if st.session_state.language == "Hindi" else 2)
        )

        # Load docs
        rows = db.execute(select(prompt_answer)).fetchall()
        docs = [Document(page_content=row.answer, metadata={"prompt": row.prompt}) for row in rows]

        if not docs:
            st.warning("No data found.")
            splits = []
            retriever = None
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever()

        st.sidebar.title("💬 Conversations")
        convs = db.execute(select(conversations).where(conversations.c.user_id == st.session_state.user_id)).fetchall()

        # Show conversations with first question as title (already saved as title)
        for conv in convs:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(conv.title, key=f"conv_{conv.id}"):
                    st.session_state.active_conversation = conv.id
                    # load conversation messages into session
                    st.session_state.messages = [
                        {"role": m.role, "content": m.content}
                        for m in db.execute(select(messages).where(messages.c.conversation_id == conv.id)).fetchall()
                    ]
            with col2:
                if st.button("🗑️", key=f"del_{conv.id}"):
                    db.execute(delete(messages).where(messages.c.conversation_id == conv.id))
                    db.execute(delete(conversations).where(conversations.c.id == conv.id))
                    db.commit()
                    if st.session_state.active_conversation == conv.id:
                        st.session_state.active_conversation = None
                        st.session_state.messages = []
                    st.success(f"Conversation '{conv.title}' deleted.")
                    st.rerun()

        if st.sidebar.button("➕ New Chat"):
            result = db.execute(insert(conversations).values(user_id=st.session_state.user_id, title="New Chat", created_at=datetime.utcnow()))
            db.commit()
            st.session_state.active_conversation = result.inserted_primary_key[0]
            st.session_state.messages = []
            st.rerun()

        if st.sidebar.button("🚪 Logout"):
            st.session_state.user_id = None
            st.session_state.active_conversation = None
            st.session_state.messages = []
            st.rerun()

        # Prepare language-aware responses
        dont_know_responses = {
            "English": "I don't know about this.",
            "Hindi": "मुझे नहीं पता।",
            "Marathi": "मला माहित नाही."
        }

        selected_lang = st.session_state.language

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

        # Display conversation messages
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        lang_placeholder = (
            "Enter your question..." if st.session_state.language == "English" else
            ("अपना प्रश्न दर्ज करें..." if st.session_state.language == "Hindi" else "आपला प्रश्न लिहा...")
        )

        # Chat input handling
        if prompt := st.chat_input(lang_placeholder):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Normalize prompt for duplicate-checking
            normalized_prompt = prompt.strip()
            normalized_key = normalized_prompt.lower()

            # Check existing prompt in DB (case-insensitive)
            rows_all = db.execute(select(prompt_answer)).fetchall()
            existing_row = None
            for r in rows_all:
                if r.prompt.strip().lower() == normalized_key:
                    existing_row = r
                    break

            # If there's no active conversation, create one and set its title to the first question
            if not st.session_state.active_conversation:
                result = db.execute(insert(conversations).values(user_id=st.session_state.user_id, title=normalized_prompt[:200], created_at=datetime.utcnow()))
                db.commit()
                st.session_state.active_conversation = result.inserted_primary_key[0]

            # If conversation exists but has title "New Chat" or empty, update it to the first question
            if st.session_state.active_conversation:
                conv_row = db.execute(select(conversations).where(conversations.c.id == st.session_state.active_conversation)).fetchone()
                if conv_row and (not conv_row.title or conv_row.title.strip().lower() == "new chat"):
                    db.execute(update(conversations).where(conversations.c.id == st.session_state.active_conversation).values(title=normalized_prompt[:200]))
                    db.commit()

            # Build docs again (fresh) so changes are picked up
            rows = db.execute(select(prompt_answer)).fetchall()
            docs = [Document(page_content=row.answer, metadata={"prompt": row.prompt}) for row in rows]
            if docs:
                embeddings = get_embeddings()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
                vectorstore = FAISS.from_documents(splits, embeddings)
                retriever = vectorstore.as_retriever()

                # rebuild rag components to pick up updated docs
                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [("system", 
                      "Given a chat history and the latest user question, formulate a standalone question that can be understood without chat history. Do NOT answer, just reformulate if needed."
                     ), MessagesPlaceholder("chat_history"), ("human", "{input}")]
                )
                history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

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

                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain, get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
            else:
                conversational_rag_chain = None

            # If exact prompt exists, just use stored answer and DO NOT save again
            if existing_row:
                answer = existing_row.answer
            else:
                # If no retriever or chain available (no docs), return dont_know
                if conversational_rag_chain is None:
                    answer = dont_know_responses[selected_lang]
                else:
                    with st.spinner("Thinking..."):
                        session_history = get_session_history(str(st.session_state.active_conversation))
                        response = conversational_rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": str(st.session_state.active_conversation)}}
                        )
                        answer = response["answer"]

                # Save new prompt->answer only if not duplicate (case-insensitive) and not empty
                if normalized_prompt and not existing_row:
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

            # Display assistant message and store in session + DB messages table
            st.chat_message("assistant").write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            if st.session_state.active_conversation:
                # store user/assistant messages in messages table
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

        db.close()
