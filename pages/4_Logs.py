from __future__ import annotations

import streamlit as st

from frontend.bootstrap import bootstrap_page
from frontend.components.chat_view import render_runtime_notices
from frontend.components.status_panels import render_logs_panel
from frontend.controller import refresh_audit_events, refresh_runtime_traces, refresh_session_snapshot


service, _config = bootstrap_page(
    "Logs",
    "Trace events, tool calls, state transitions, and frontend activity.",
)

render_runtime_notices(st.session_state)

if not st.session_state.get("audit_events"):
    with st.spinner("Loading audit events..."):
        refresh_audit_events(st.session_state, service)
if not st.session_state.get("runtime_traces"):
    with st.spinner("Loading runtime traces..."):
        refresh_runtime_traces(st.session_state, service)

st.write(
    "This page provides the evidence trail for a run: traces, tool-level output, state transitions, "
    "backend audit records, and the local frontend activity log."
)

controls = st.columns(3)
if controls[0].button("Refresh session snapshot", use_container_width=True):
    with st.spinner("Refreshing session snapshot..."):
        refresh_session_snapshot(st.session_state, service)
    st.rerun()
if controls[1].button("Refresh audit trail", use_container_width=True):
    with st.spinner("Refreshing audit trail..."):
        refresh_audit_events(st.session_state, service)
    st.rerun()
if controls[2].button("Refresh runtime traces", use_container_width=True):
    with st.spinner("Refreshing runtime traces..."):
        refresh_runtime_traces(st.session_state, service)
    st.rerun()

render_logs_panel(
    st.session_state.get("last_interaction"),
    st.session_state.get("activity_log", []),
    st.session_state.get("audit_events", []),
    st.session_state.get("runtime_traces", []),
)
