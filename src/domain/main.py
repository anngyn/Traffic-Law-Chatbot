# main.py — Streamlit UI (thin client calling the FastAPI backend).
from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Chatbot Luật Giao Thông", layout="centered")
st.title("Hệ thống truy vấn thông tin Luật Giao Thông Việt Nam")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào bạn! Mình là trợ lý hỗ trợ tìm kiếm thông tin về luật giao thông Việt Nam. Bạn cần mình giúp điều gì không?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Bạn muốn hỏi gì về luật giao thông?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm thông tin..."):
            try:
                resp = requests.post(f"{API_URL}/chat", json={"message": user_input}, timeout=120)
                resp.raise_for_status()
                answer = resp.json()["answer"]
            except Exception as e:
                answer = f"[Lỗi kết nối API]: {e}"
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
