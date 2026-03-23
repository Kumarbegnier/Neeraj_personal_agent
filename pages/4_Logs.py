from __future__ import annotations

import streamlit as st

from frontend.bootstrap import bootstrap_page
from frontend.components.chat_view import render_runtime_notices
from frontend.components.status_panels import render_logs_panel
from frontend.controller import refresh_audit_events, refresh_session_snapshot


service, _config = bootstrap_page(
    "Logs",
    "Trace events, tool calls, state transitions, and frontend activity.",
)

render_runtime_notices(st.session_state)

if not st.session_state.get("audit_events"):
    refresh_audit_events(st.session_state, service)

controls = st.columns(2)
if controls[0].button("Refresh session snapshot", use_container_width=True):
    refresh_session_snapshot(st.session_state, service)
    st.rerun()
if controls[1].button("Refresh audit trail", use_container_width=True):
    refresh_audit_events(st.session_state, service)
    st.rerun()

render_logs_panel(
    st.session_state.get("last_interaction"),
    st.session_state.get("activity_log", []),
    st.session_state.get("audit_events", []),
)
