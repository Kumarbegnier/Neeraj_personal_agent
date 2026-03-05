from __future__ import annotations

import os
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = Path(__file__).resolve().parent / "assets"
FLASHSPACE_AVATAR = str(ASSETS_ROOT / "flashspace_face.svg")
USER_AVATAR = str(ASSETS_ROOT / "user_face.svg")
load_dotenv(BACKEND_ROOT / ".env")


def _default_conversation_id() -> str:
    return os.getenv("CLI_CONVERSATION_ID") or "default"


def _default_base_url() -> str:
    return (os.getenv("CLI_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")


def _normalize_token(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    parts = value.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return value


def _call_chat_api(*, base_url: str, token: str, query: str, conversation_id: str, session_id: str | None):
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload: dict[str, str] = {"query": query, "conversation_id": conversation_id}
    if session_id:
        payload["session_id"] = session_id

    resp = requests.post(f"{base_url}/chat", headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()


def _inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&display=swap');
        :root {
            --fs-bg-soft: #f8fbff;
            --fs-text-main: #0f172a;
            --fs-text-muted: #4b5563;
            --fs-brand-blue: #2563eb;
            --fs-brand-cyan: #22c1a6;
            --fs-card-border: rgba(255, 255, 255, 0.72);
            --fs-user-bubble: linear-gradient(135deg, rgba(233, 246, 255, 0.92), rgba(224, 236, 255, 0.9));
            --fs-bot-bubble: linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(250, 243, 255, 0.9));
        }
        .stApp {
            font-family: 'Inter', sans-serif;
            background:
                radial-gradient(circle at 18% 34%, rgba(53, 111, 252, 0.42), rgba(53, 111, 252, 0) 38%),
                radial-gradient(circle at 80% 38%, rgba(64, 108, 247, 0.45), rgba(64, 108, 247, 0) 36%),
                radial-gradient(circle at 50% 100%, rgba(255, 0, 132, 0.65), rgba(255, 0, 132, 0) 48%),
                linear-gradient(180deg, #f8fafc 0%, #e9f0ff 44%, #ffd5ef 100%);
            min-height: 100vh;
        }
        .block-container {
            padding-top: 0.55rem;
            max-width: 1180px;
        }
        [data-testid="stSidebar"] {
            background: rgba(245, 247, 250, 0.9);
            border-right: 1px solid #e5e7eb;
        }
        .fs-topbar {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            box-shadow: 0 8px 26px rgba(17, 24, 39, 0.08);
        }
        .fs-brand {
            font-size: 1.8rem;
            font-weight: 900;
            color: #111827;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .fs-brand-face {
            font-size: 0.8rem;
            width: 26px;
            height: 26px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-weight: 800;
            background: linear-gradient(135deg, #4f46e5, #ec4899);
        }
        .fs-brand span {
            color: #2ca59b;
            font-style: italic;
        }
        .fs-nav a {
            text-decoration: none;
            color: #374151;
            font-size: 0.95rem;
            font-weight: 600;
            margin-left: 18px;
        }
        .fs-nav a:hover { color: #111827; }
        .fs-nav .btn-light {
            border: 1px solid #d1d5db;
            border-radius: 10px;
            padding: 6px 12px;
            background: #ffffff;
        }
        .fs-nav .btn-dark {
            border-radius: 10px;
            padding: 7px 13px;
            background: #111827;
            color: #ffffff !important;
        }
        .fs-hero {
            text-align: center;
            padding: 4rem 1rem 2rem 1rem;
        }
        .fs-title {
            font-size: 3.1rem;
            font-weight: 800;
            color: #101828;
            margin-bottom: 0.3rem;
            letter-spacing: -0.6px;
        }
        .fs-subtitle {
            font-size: 1.45rem;
            color: #22a3a3;
            font-weight: 800;
            font-style: italic;
            margin-bottom: 0.75rem;
        }
        .fs-desc {
            font-size: 1.02rem;
            color: #4b5563;
            max-width: 760px;
            margin: 0 auto 1.1rem auto;
            line-height: 1.45;
        }
        .fs-role-wrap {
            margin-bottom: 0.9rem;
        }
        .fs-role-chip {
            display: inline-block;
            font-size: 0.8rem;
            font-weight: 700;
            border-radius: 999px;
            padding: 0.28rem 0.75rem;
            margin: 0 0.22rem;
            color: #374151;
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid #d1d5db;
        }
        .fs-role-chip.active {
            color: #ffffff;
            border-color: #111827;
            background: #111827;
        }
        .fs-prompt-box {
            margin: 0 auto 1.4rem auto;
            width: min(820px, 95%);
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(245,247,255,0.9));
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,0.75);
            box-shadow: 0 18px 38px rgba(17, 24, 39, 0.15), 0 0 0 1px rgba(110, 99, 255, 0.1) inset;
            text-align: left;
            padding: 1rem 1.05rem;
            position: relative;
            overflow: hidden;
        }
        .fs-prompt-box::before {
            content: "";
            position: absolute;
            inset: -1px;
            border-radius: 24px;
            padding: 1px;
            background: linear-gradient(120deg, rgba(104,93,255,0.35), rgba(236,72,153,0.28), rgba(59,130,246,0.3));
            -webkit-mask:
                linear-gradient(#fff 0 0) content-box,
                linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
        }
        .fs-prompt-title {
            color: #4b5563;
            font-size: 1.03rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        .fs-prompt-hint {
            color: #6b7280;
            font-size: 0.88rem;
            margin-bottom: 0.75rem;
        }
        .fs-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 0.75rem;
        }
        .fs-chip {
            font-size: 0.78rem;
            font-weight: 600;
            color: #374151;
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid #dbe2ff;
            border-radius: 999px;
            padding: 4px 10px;
        }
        .fs-prompt-footer {
            display: flex;
            justify-content: space-between;
            color: #6b7280;
            font-size: 0.9rem;
        }
        .fs-main-tabs {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }
        .fs-main-tab {
            background: rgba(255,255,255,0.82);
            border: 1px solid #d3dbef;
            border-radius: 14px;
            padding: 10px 14px;
            font-size: 0.9rem;
            font-weight: 700;
            color: #374151;
        }
        .fs-main-tab.active {
            border-color: #22a3a3;
            color: #0f766e;
            box-shadow: 0 6px 18px rgba(34,163,163,0.18);
        }
        .fs-solution-grid {
            max-width: 1040px;
            margin: 0 auto 1.6rem auto;
            display: grid;
            grid-template-columns: repeat(3, minmax(220px, 1fr));
            gap: 12px;
        }
        .fs-solution-card {
            background: rgba(255,255,255,0.85);
            border: 1px solid #dbe5f8;
            border-radius: 16px;
            padding: 12px;
            box-shadow: 0 8px 20px rgba(17, 24, 39, 0.08);
        }
        .fs-solution-title {
            font-size: 0.96rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 6px;
        }
        .fs-solution-copy {
            font-size: 0.86rem;
            color: #4b5563;
            line-height: 1.35;
            margin-bottom: 8px;
        }
        .fs-solution-link {
            font-size: 0.82rem;
            color: #0f766e;
            font-weight: 700;
        }
        .fs-stats {
            max-width: 1040px;
            margin: 0 auto 1.4rem auto;
            display: grid;
            grid-template-columns: repeat(4, minmax(120px, 1fr));
            gap: 10px;
        }
        .fs-stat {
            text-align: center;
            background: rgba(255,255,255,0.75);
            border: 1px solid #dbe5f8;
            border-radius: 14px;
            padding: 10px 8px;
        }
        .fs-stat-value {
            font-size: 1.15rem;
            font-weight: 900;
            color: #111827;
        }
        .fs-stat-label {
            font-size: 0.78rem;
            color: #6b7280;
            font-weight: 600;
        }
        .fs-chat-shell {
            margin: 0 auto;
            width: min(940px, 98%);
            padding: 0.45rem;
            border-radius: 22px;
            background: linear-gradient(145deg, rgba(255,255,255,0.34), rgba(255,255,255,0.16));
            border: 1px solid var(--fs-card-border);
            box-shadow: 0 22px 48px rgba(30, 64, 175, 0.12);
        }
        .fs-cute-center {
            display: flex;
            justify-content: center;
            margin: 0.2rem 0 1rem 0;
        }
        .fs-cute-orb {
            position: relative;
            width: 112px;
            height: 112px;
            border-radius: 999px;
            background: radial-gradient(circle at 28% 28%, #ffffff 8%, #dbeafe 36%, #93c5fd 68%, #60a5fa 100%);
            border: 2px solid rgba(255, 255, 255, 0.9);
            box-shadow: 0 18px 38px rgba(59, 130, 246, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fsFloat 2.8s ease-in-out infinite, fsPulse 2.2s ease-in-out infinite;
        }
        .fs-cute-face {
            width: 72px;
            height: 72px;
            border-radius: 999px;
            background: linear-gradient(135deg, #2563eb, #22c1a6);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 0.84rem;
            font-weight: 900;
            letter-spacing: 0.3px;
            text-transform: lowercase;
        }
        .fs-cute-face em {
            font-style: italic;
            color: #d7fff7;
        }
        .fs-spark {
            position: absolute;
            color: #ffffff;
            font-size: 0.95rem;
            animation: fsBlink 1.9s ease-in-out infinite;
        }
        .fs-spark.left { top: 8px; left: 14px; }
        .fs-spark.right { top: 16px; right: 12px; animation-delay: 0.5s; }
        @keyframes fsFloat {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
        }
        @keyframes fsPulse {
            0%, 100% { box-shadow: 0 18px 38px rgba(59, 130, 246, 0.3); }
            50% { box-shadow: 0 22px 46px rgba(34, 193, 166, 0.34); }
        }
        @keyframes fsBlink {
            0%, 100% { opacity: 0.5; transform: scale(0.9); }
            50% { opacity: 1; transform: scale(1.15); }
        }
        [data-testid="stChatMessage"] {
            border-radius: 18px;
            border: 1px solid var(--fs-card-border);
            background: var(--fs-bot-bubble);
            backdrop-filter: blur(6px);
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.08);
            padding: 0.58rem 0.68rem;
            margin-bottom: 0.72rem;
        }
        [data-testid="stChatMessageAvatarUser"] img,
        [data-testid="stChatMessageAvatarAssistant"] img {
            border-radius: 999px !important;
            border: 2px solid rgba(255,255,255,0.82);
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15);
        }
        [data-testid="stChatMessageContent"] p {
            color: var(--fs-text-main);
            line-height: 1.48;
            font-size: 0.98rem;
        }
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: var(--fs-user-bubble);
            border-color: rgba(191, 219, 254, 0.9);
        }
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            border-color: rgba(233, 213, 255, 0.85);
        }
        [data-testid="stChatInput"] input {
            min-height: 52px !important;
            border-radius: 16px !important;
            border: 1px solid #cbd5e1 !important;
            background: rgba(255, 255, 255, 0.98) !important;
            box-shadow: 0 10px 20px rgba(59, 130, 246, 0.12);
        }
        [data-testid="stChatInput"] input:focus {
            border-color: #60a5fa !important;
            box-shadow: 0 0 0 3px rgba(147, 197, 253, 0.35), 0 12px 24px rgba(59, 130, 246, 0.16) !important;
        }
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #d1d5db;
            color: #111827;
            background: #ffffff;
            font-weight: 600;
        }
        .stButton > button:hover {
            border-color: #9ca3af;
            color: #111827;
        }
        @media (max-width: 900px) {
            .fs-nav a { display: none; }
            .fs-nav a.btn-light, .fs-nav a.btn-dark { display: inline-block; }
            .fs-title { font-size: 2.15rem; }
            .fs-subtitle { font-size: 1.08rem; }
            .fs-hero { padding-top: 3.2rem; }
            .fs-solution-grid { grid-template-columns: 1fr; }
            .fs-stats { grid-template-columns: repeat(2, 1fr); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _role_line(active_role: str) -> str:
    roles = ["admin", "guest", "user", "affiliate", "partner"]
    chips: list[str] = []
    for role in roles:
        classes = "fs-role-chip active" if role == active_role else "fs-role-chip"
        chips.append(f'<span class="{classes}">{role.upper()}</span>')
    return "".join(chips)


def _render_sidebar():
    st.sidebar.header("Connection")
    st.session_state.base_url = st.sidebar.text_input("Backend URL", st.session_state.base_url).rstrip("/")
    st.session_state.jwt = st.sidebar.text_input("JWT Token", st.session_state.jwt, type="password")
    st.session_state.conversation_id = st.sidebar.text_input("Conversation ID", st.session_state.conversation_id)
    st.session_state.role_mode = st.sidebar.selectbox(
        "Role Mode",
        options=["admin", "guest", "user", "affiliate", "partner"],
        index=["admin", "guest", "user", "affiliate", "partner"].index(st.session_state.role_mode),
    )
    st.session_state.session_id = st.sidebar.text_input(
        "Session ID (optional)", st.session_state.session_id, placeholder="auto-generated by backend"
    )

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = ""
            st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            st.rerun()
    with c2:
        if st.button("Ping Health", use_container_width=True):
            try:
                r = requests.get(f"{st.session_state.base_url}/health", timeout=10)
                r.raise_for_status()
                st.sidebar.success(f"Healthy: {r.json()}")
            except Exception as exc:
                st.sidebar.error(f"Health check failed: {exc}")


def _init_state():
    st.session_state.setdefault("base_url", _default_base_url())
    st.session_state.setdefault("jwt", os.getenv("CLI_JWT", ""))
    st.session_state.setdefault("conversation_id", _default_conversation_id())
    st.session_state.setdefault("session_id", os.getenv("CLI_SESSION_ID", ""))
    st.session_state.setdefault("role_mode", "user")
    if st.session_state.role_mode not in {"admin", "guest", "user", "affiliate", "partner"}:
        st.session_state.role_mode = "user"
    st.session_state.setdefault("messages", [])


def main():
    st.set_page_config(page_title="FlashSpace AI Chat", page_icon="chat", layout="wide")
    _init_state()
    _inject_css()
    _render_sidebar()

    st.markdown(
        """
        <div class="fs-topbar">
          <div class="fs-brand"><span class="fs-brand-face">FS</span>flash<span>space</span></div>
          <div class="fs-nav">
            <a href="#">Solutions</a>
            <a href="#">Resources</a>
            <a href="#">Enterprise</a>
            <a href="#">Pricing</a>
            <a href="#">Community</a>
            <a class="btn-light" href="#">Log in</a>
            <a class="btn-dark" href="#">Get started</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="fs-hero">
          <div class="fs-title">FlashSpace Chatbot</div>
          <div class="fs-subtitle">Fast Support</div>
          <div class="fs-desc">Ask anything about workspaces, bookings, payments, invoices, and support.</div>
          <div class="fs-role-wrap">{_role_line(st.session_state.role_mode)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="fs-cute-center">
          <div class="fs-cute-orb">
            <span class="fs-spark left">✦</span>
            <span class="fs-spark right">✦</span>
            <div class="fs-cute-face">flash <em>space</em></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fs-chat-shell">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else FLASHSPACE_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Type your message...")
    if not prompt:
        return

    token = _normalize_token(st.session_state.jwt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=FLASHSPACE_AVATAR):
        with st.spinner("Thinking..."):
            try:
                role_hint = st.session_state.role_mode
                query_with_role = f"[role_context={role_hint}] {prompt}"
                data = _call_chat_api(
                    base_url=st.session_state.base_url,
                    token=token,
                    query=query_with_role,
                    conversation_id=st.session_state.conversation_id.strip() or "default",
                    session_id=st.session_state.session_id.strip() or None,
                )
                reply = str(data.get("reply", "")).strip() or "No reply received."
                new_session = str(data.get("session_id", "")).strip()
                if new_session:
                    st.session_state.session_id = new_session
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except requests.HTTPError as exc:
                body = exc.response.text if exc.response is not None else ""
                err = f"HTTP error: {exc}. {body}".strip()
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as exc:
                err = f"Request failed: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()
