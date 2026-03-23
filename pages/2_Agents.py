from __future__ import annotations

import streamlit as st

from frontend.bootstrap import bootstrap_page
from frontend.components.chat_view import render_runtime_notices
from frontend.components.status_panels import (
    render_agent_review_panels,
    render_execution_panel,
    render_plan_panel,
    render_runtime_catalog_panel,
)
from frontend.controller import preview_plan, refresh_runtime_catalogs, refresh_session_snapshot


service, _config = bootstrap_page(
    "Agents",
    "Detailed planner, routing, verification, and execution inspection.",
)

render_runtime_notices(st.session_state)

if st.session_state.get("agent_catalog") is None or st.session_state.get("tool_catalog") is None:
    refresh_runtime_catalogs(st.session_state, service)

controls = st.columns(3)
if controls[0].button("Refresh session", use_container_width=True):
    refresh_session_snapshot(st.session_state, service)
    st.rerun()
if controls[1].button(
    "Preview last plan",
    use_container_width=True,
    disabled=not bool(st.session_state.get("last_prompt")),
):
    preview_plan(st.session_state, service, str(st.session_state["last_prompt"]))
    st.rerun()
if controls[2].button("Refresh catalogs", use_container_width=True):
    refresh_runtime_catalogs(st.session_state, service)
    st.rerun()

interaction = st.session_state.get("last_interaction")
plan_preview = st.session_state.get("last_plan")

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    render_plan_panel(interaction, plan_preview)
with right:
    render_execution_panel(interaction)

render_runtime_catalog_panel(
    st.session_state.get("agent_catalog"),
    st.session_state.get("tool_catalog"),
)
render_agent_review_panels(interaction, plan_preview)
