import streamlit as st
from sqlalchemy import create_engine, Table, Column, Integer, String, DateTime, Text, MetaData, select, insert, update
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# ------------------------
# Database Setup (same as main.py)
# ------------------------
engine = create_engine("sqlite:///rag_chat_app.db", connect_args={"check_same_thread": False})
metadata = MetaData()

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
# Streamlit Page Config
# ------------------------
st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("🧠 Admin Dashboard – Manage Prompt & Answer Database")

# ------------------------
# Admin Authentication
# ------------------------
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if not st.session_state.is_admin:
    st.subheader("🔐 Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "Patil" and password == "000":
            st.session_state.is_admin = True
            st.success("✅ Admin login successful!")
            st.rerun()
        else:
            st.error("Invalid admin credentials!")

# ------------------------
# Main Dashboard (after login)
# ------------------------
else:
    db = SessionLocal()

    st.sidebar.title("⚙️ Actions")
    action = st.sidebar.radio("Select Action:", ["View Database", "Add or Update Answer", "Logout"])

    # ------------------------
    # View Database Table
    # ------------------------
    if action == "View Database":
        st.subheader("📋 Prompt–Answer Table")
        data = db.execute(select(prompt_answer)).fetchall()

        if not data:
            st.warning("No records found in the database.")
        else:
            st.dataframe(
                [{"ID": row.id, "Prompt": row.prompt, "Answer": row.answer, "Timestamp": row.timestamp} for row in data]
            )

    # ------------------------
    # Add or Update Answer
    # ------------------------
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

    # ------------------------
    # Logout
    # ------------------------
    elif action == "Logout":
        st.session_state.is_admin = False
        st.success("Logged out successfully!")
        st.rerun()

    db.close()
