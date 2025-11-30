import streamlit as st
import uuid
import re
import json
import os

STORAGE_FILE = "chat_history.json"


def save_chats_to_file():
    with open(STORAGE_FILE, "w") as f:
        json.dump(st.session_state["chats"], f)


def load_chats_from_file():
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, "r") as f:
            return json.load(f)
    return {}


def initialize_chat_system():
    if "chats" not in st.session_state:
        st.session_state["chats"] = load_chats_from_file()
    if "current_chat_id" not in st.session_state:
        st.session_state["current_chat_id"] = None
    if "active_menu" not in st.session_state:
        st.session_state["active_menu"] = None


def clean_title(text):
    text = re.sub(r"[^\w\s]", "", text).lower()

    stopwords = {
        "how",
        "what",
        "why",
        "when",
        "where",
        "does",
        "do",
        "is",
        "are",
        "can",
        "tell",
        "me",
        "the",
        "a",
        "an",
        "to",
        "in",
        "on",
        "explain",
        "describe",
        "define",
        "give",
        "information",
        "explain",
    }

    words = [w for w in text.split() if w not in stopwords]

    meaningful = words[:6]

    title = " ".join(meaningful).title()

    return title if title.strip() else "Medical Chat"


def create_new_chat(initial_title=None):
    title = clean_title(initial_title) if initial_title else "New Chat"
    chat_id = str(uuid.uuid4())
    st.session_state["chats"][chat_id] = {"title": title, "messages": []}
    st.session_state["current_chat_id"] = chat_id
    save_chats_to_file()
    return chat_id


def delete_chat(chat_id):
    if chat_id in st.session_state["chats"]:
        del st.session_state["chats"][chat_id]

    st.session_state["active_menu"] = None
    save_chats_to_file()

    if st.session_state["chats"]:
        st.session_state["current_chat_id"] = list(st.session_state["chats"].keys())[-1]
    else:
        st.session_state["current_chat_id"] = None


def rename_chat(chat_id, new_title):
    st.session_state["chats"][chat_id]["title"] = new_title
    save_chats_to_file()


def get_current_chat_messages():
    chat_id = st.session_state.get("current_chat_id")
    if chat_id and chat_id in st.session_state["chats"]:
        return st.session_state["chats"][chat_id]["messages"]
    return []


def add_message_to_current_chat(role, content):
    if st.session_state["current_chat_id"] is None:
        chat_id = create_new_chat(initial_title=content)
        st.session_state["current_chat_id"] = chat_id

    st.session_state["chats"][st.session_state["current_chat_id"]]["messages"].append(
        {"role": role, "content": content}
    )

    save_chats_to_file()


def sidebar_chats():
    st.title("💬 Your Chats")

    if st.button("➕ New Chat"):
        st.session_state["current_chat_id"] = None
        st.rerun()

    for chat_id, chat in st.session_state["chats"].items():
        cols = st.columns([7, 1])

        if cols[0].button(chat["title"], key=f"select_{chat_id}"):
            st.session_state["current_chat_id"] = chat_id

        if cols[1].button("⋮", key=f"menu_{chat_id}"):
            st.session_state["active_menu"] = (
                None if st.session_state["active_menu"] == chat_id else chat_id
            )

        if st.session_state["active_menu"] == chat_id:
            new_title = st.text_input(
                "Rename Chat", value=chat["title"], key=f"rename_{chat_id}"
            )

            if st.button("Save", key=f"save_{chat_id}"):
                rename_chat(chat_id, new_title)
                st.session_state["active_menu"] = None

            if st.button("Delete", key=f"delete_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()

            st.write("---")
