from __future__ import annotations

import streamlit as st

from frontend.bootstrap import bootstrap_page
from frontend.components.chat_view import render_runtime_notices
from frontend.components.status_panels import render_home_dashboard
from frontend.controller import (
    refresh_architecture,
    refresh_health,
    refresh_runtime_catalogs,
    refresh_session_snapshot,
)


service, _config = bootstrap_page(
    "Neeraj AI OS",
    "Research console for planning, routing, memory, execution, and audit inspection.",
)

if st.session_state["last_health"] is None:
    with st.spinner("Connecting to the FastAPI backend..."):
        refresh_health(st.session_state, service)
if not st.session_state["architecture"]:
    with st.spinner("Loading runtime architecture..."):
        refresh_architecture(st.session_state, service)
if st.session_state.get("agent_catalog") is None or st.session_state.get("tool_catalog") is None:
    with st.spinner("Loading runtime catalogs..."):
        refresh_runtime_catalogs(st.session_state, service)

render_runtime_notices(st.session_state)

st.write(
    "Use this workspace as the control room for the Neeraj AI OS runtime: monitor backend health, "
    "inspect the active architecture, and move into the specialized pages for chat, planning, memory, logs, runtime intelligence, and settings."
)

navigation = st.columns(6)
navigation[0].page_link("pages/1_Chat.py", label="Chat Workspace", use_container_width=True)
navigation[1].page_link("pages/2_Agents.py", label="Agents", use_container_width=True)
navigation[2].page_link("pages/3_Memory.py", label="Memory", use_container_width=True)
navigation[3].page_link("pages/4_Logs.py", label="Logs", use_container_width=True)
navigation[4].page_link("pages/6_Intelligence.py", label="Intelligence", use_container_width=True)
navigation[5].page_link("pages/5_Settings.py", label="Settings", use_container_width=True)

actions = st.columns(4)
if actions[0].button("Refresh backend health", use_container_width=True):
    with st.spinner("Refreshing backend health..."):
        refresh_health(st.session_state, service)
    st.rerun()
if actions[1].button("Refresh session snapshot", use_container_width=True):
    with st.spinner("Refreshing session snapshot..."):
        refresh_session_snapshot(st.session_state, service)
    st.rerun()
if actions[2].button("Refresh architecture map", use_container_width=True):
    with st.spinner("Refreshing architecture map..."):
        refresh_architecture(st.session_state, service)
    st.rerun()
if actions[3].button("Refresh catalogs", use_container_width=True):
    with st.spinner("Refreshing runtime catalogs..."):
        refresh_runtime_catalogs(st.session_state, service)
    st.rerun()

render_home_dashboard(
    st.session_state,
    st.session_state.get("last_health"),
    st.session_state.get("architecture", []),
)
