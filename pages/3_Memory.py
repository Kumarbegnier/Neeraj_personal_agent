from __future__ import annotations

import streamlit as st

from frontend.bootstrap import bootstrap_page
from frontend.components.chat_view import render_runtime_notices
from frontend.components.status_panels import render_memory_collections, render_memory_panel
from frontend.controller import refresh_session_snapshot
from frontend.utils.state import current_memory


service, _config = bootstrap_page(
    "Memory",
    "Working memory, retrieved evidence, and durable session context.",
)

render_runtime_notices(st.session_state)

if st.button("Refresh session snapshot", use_container_width=True):
    refresh_session_snapshot(st.session_state, service)
    st.rerun()

memory = current_memory(st.session_state)
session_snapshot = st.session_state.get("last_session_snapshot")

render_memory_panel(memory, session_snapshot)
render_memory_collections(memory)
