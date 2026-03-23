from __future__ import annotations

import streamlit as st

from frontend.bootstrap import bootstrap_page
from frontend.components.chat_view import render_runtime_notices
from frontend.components.status_panels import render_home_dashboard
from frontend.controller import refresh_architecture, refresh_health, refresh_session_snapshot


service, _config = bootstrap_page(
    "Neeraj AI OS",
    "Research-grade frontend for the stateful cognitive runtime.",
)

if st.session_state["last_health"] is None:
    refresh_health(st.session_state, service)
if not st.session_state["architecture"]:
    refresh_architecture(st.session_state, service)

render_runtime_notices(st.session_state)

actions = st.columns(3)
if actions[0].button("Refresh backend health", use_container_width=True):
    refresh_health(st.session_state, service)
    st.rerun()
if actions[1].button("Refresh session snapshot", use_container_width=True):
    refresh_session_snapshot(st.session_state, service)
    st.rerun()
if actions[2].button("Refresh architecture map", use_container_width=True):
    refresh_architecture(st.session_state, service)
    st.rerun()

render_home_dashboard(
    st.session_state,
    st.session_state.get("last_health"),
    st.session_state.get("architecture", []),
)
