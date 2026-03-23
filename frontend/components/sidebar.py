from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import streamlit as st

from frontend.config import FrontendConfig
from frontend.view_models import humanize_label, selected_agent_label


def render_sidebar(state: MutableMapping[str, Any], config: FrontendConfig) -> None:
    catalog = state.get("agent_catalog")
    selected_agent = selected_agent_label(
        str(state["selected_agent"]),
        getattr(catalog, "agents", None),
    )
    health = state.get("last_health")

    with st.sidebar:
        st.markdown("## Neeraj AI OS")
        st.caption("Research-grade control surface for the stateful runtime.")

        with st.container(border=True):
            st.markdown("**Current Session**")
            st.caption(f"User: `{state['user_id']}`")
            st.caption(f"Session: `{state['session_id']}`")
            st.caption(f"Backend: `{state['backend_url']}`")
            if health is not None:
                st.caption(f"Backend status: `{humanize_label(health.status)}`")

        with st.container(border=True):
            st.markdown("**Runtime Controls**")
            st.selectbox(
                "Model selection",
                options=list(config.model_options),
                key="selected_model",
                help="Attached to outbound requests as frontend metadata and backend preferences.",
            )
            st.toggle(
                "Auto-approve gated actions",
                key="approval_toggle",
                help="When enabled, the frontend sends approval metadata for requests that may require confirmation.",
            )

        with st.container(border=True):
            st.markdown("**Active Runtime State**")
            metrics_top = st.columns(2)
            metrics_bottom = st.columns(2)
            metrics_top[0].metric("Selected agent", selected_agent)
            metrics_top[1].metric("Agent status", humanize_label(str(state["agent_status"])))
            metrics_bottom[0].metric("Risk level", humanize_label(str(state["risk_level"])))
            metrics_bottom[1].metric("Approval", "Required" if state["approval_required"] else "Clear")

        with st.container(border=True):
            st.markdown("**Memory Summary**")
            st.text_area(
                "memory_summary",
                value=str(state["memory_summary"]),
                disabled=True,
                height=180,
                label_visibility="collapsed",
            )

        if state.get("pending_prompt"):
            with st.container(border=True):
                st.markdown("**Approval Queue**")
                st.caption("The latest request is waiting on explicit approval before side effects can proceed.")
                st.write(str(state["pending_prompt"]))
