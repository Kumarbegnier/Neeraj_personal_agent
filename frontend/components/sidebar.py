from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import streamlit as st

from frontend.config import FrontendConfig


def render_sidebar(state: MutableMapping[str, Any], config: FrontendConfig) -> None:
    selected_agent = str(state["selected_agent"])
    catalog = state.get("agent_catalog")
    if catalog is not None:
        for descriptor in getattr(catalog, "agents", []):
            if getattr(descriptor, "key", None) == selected_agent:
                selected_agent = str(getattr(descriptor, "display_name", selected_agent))
                break

    with st.sidebar:
        st.markdown("## Runtime Console")
        st.caption(config.app_title)

        st.markdown("### Session")
        st.caption(f"User: `{state['user_id']}`")
        st.caption(f"Session: `{state['session_id']}`")
        st.caption(f"Backend: `{state['backend_url']}`")

        st.markdown("### Controls")
        st.selectbox(
            "Model selection",
            options=list(config.model_options),
            key="selected_model",
            help="This selection is attached to requests as frontend metadata and preferences.",
        )
        st.toggle(
            "Auto-approve gated actions",
            key="approval_toggle",
            help="When enabled, the frontend sends explicit approval metadata for sensitive requests.",
        )

        st.markdown("### Runtime State")
        st.metric("Selected agent", selected_agent)
        st.metric("Agent status", str(state["agent_status"]).replace("_", " ").title())
        st.metric("Risk level", str(state["risk_level"]).upper())
        st.metric("Approval needed", "YES" if state["approval_required"] else "NO")

        st.markdown("### Memory Summary")
        st.text_area(
            "memory",
            value=str(state["memory_summary"]),
            disabled=True,
            height=180,
            label_visibility="collapsed",
        )
