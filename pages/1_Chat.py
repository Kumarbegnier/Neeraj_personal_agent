from __future__ import annotations

import streamlit as st

from frontend.bootstrap import bootstrap_page
from frontend.components.chat_view import render_chat_workspace, render_runtime_notices
from frontend.components.status_panels import (
    render_execution_panel,
    render_logs_panel,
    render_memory_panel,
    render_plan_panel,
)
from frontend.controller import (
    approve_pending_request,
    clear_workspace,
    preview_plan,
    refresh_session_snapshot,
    submit_chat_prompt,
)
from frontend.utils.state import current_memory


service, _config = bootstrap_page(
    "Chat Workspace",
    "Primary research console for routed chat, plan previews, memory, execution, and audit visibility.",
)

render_runtime_notices(st.session_state)

st.write(
    "Use the chat surface for serious, instrumented conversations with the backend runtime. "
    "The diagnostics panel keeps the planning, memory, execution, and log views attached to the active session."
)

toolbar = st.columns(4)
if toolbar[0].button("Refresh session", use_container_width=True):
    with st.spinner("Refreshing session snapshot..."):
        refresh_session_snapshot(st.session_state, service)
    st.rerun()
if toolbar[1].button(
    "Preview last plan",
    use_container_width=True,
    disabled=not bool(st.session_state.get("last_prompt")),
):
    with st.spinner("Generating planner preview..."):
        preview_plan(st.session_state, service, str(st.session_state["last_prompt"]))
    st.rerun()
if toolbar[2].button(
    "Approve pending",
    use_container_width=True,
    disabled=not bool(st.session_state.get("pending_prompt")),
):
    with st.spinner("Re-executing with explicit approval..."):
        approve_pending_request(st.session_state, service)
    st.rerun()
if toolbar[3].button("Clear workspace", use_container_width=True):
    clear_workspace(st.session_state)
    st.rerun()

workspace, diagnostics = st.columns([1.35, 1.0], gap="large")

with workspace:
    prompt = render_chat_workspace(
        st.session_state["messages"],
        prompt_placeholder="Submit a prompt to Neeraj AI OS",
    )
    if prompt:
        with st.spinner("Running the orchestrated agent loop..."):
            submit_chat_prompt(st.session_state, service, prompt)
        st.rerun()

with diagnostics:
    tabs = st.tabs(["Planner", "Memory", "Execution", "Logs"])
    with tabs[0]:
        render_plan_panel(
            st.session_state.get("last_interaction"),
            st.session_state.get("last_plan"),
        )
    with tabs[1]:
        render_memory_panel(
            current_memory(st.session_state),
            st.session_state.get("last_session_snapshot"),
        )
    with tabs[2]:
        render_execution_panel(st.session_state.get("last_interaction"))
    with tabs[3]:
        render_logs_panel(
            st.session_state.get("last_interaction"),
            st.session_state.get("activity_log", []),
            st.session_state.get("audit_events", []),
        )
