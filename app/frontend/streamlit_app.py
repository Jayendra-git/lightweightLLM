from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.shared.config import DEFAULT_API_URL


st.set_page_config(page_title="PersonaTutor", page_icon="💬", layout="wide")
st.title("PersonaTutor")
st.caption("A tiny local demo: LoRA style tuning + RAG over hand-written docs.")

api_url = st.sidebar.text_input("Backend URL", value=DEFAULT_API_URL)
top_k = st.sidebar.slider("Retrieved chunks", min_value=1, max_value=5, value=3)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about LoRA, RAG, or related concepts")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        response = requests.post(
            f"{api_url}/chat",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        payload = {
            "answer": f"Backend request failed: {exc}",
            "adapter_loaded": False,
            "sources": [],
        }

    with st.chat_message("assistant"):
        st.markdown(payload["answer"])
        badge = "LoRA adapter loaded" if payload["adapter_loaded"] else "Base model only"
        st.caption(badge)
        if payload["sources"]:
            st.markdown("**Retrieved sources**")
            for source in payload["sources"]:
                with st.expander(source["source"]):
                    st.write(source["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": payload["answer"]}
    )
