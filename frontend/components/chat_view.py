from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import streamlit as st

from frontend.components.status_panels import render_empty_state
from frontend.utils.state import TranscriptEntry


def render_runtime_notices(state: MutableMapping[str, Any]) -> None:
    if state.get("last_error"):
        st.error(str(state["last_error"]))
    if state.get("approval_required") and state.get("pending_prompt"):
        st.warning("The latest request is waiting on approval before side effects can proceed.")


def render_chat_workspace(
    messages: Sequence[TranscriptEntry],
    *,
    prompt_placeholder: str,
) -> str | None:
    if not messages:
        render_empty_state(
            "Chat workspace is ready.",
            "Submit a prompt to start the research and execution loop.",
        )
    else:
        for entry in messages:
            with st.chat_message(entry.role):
                st.markdown(entry.content)
                if entry.caption:
                    st.caption(f"{entry.caption} | {entry.timestamp}")

    return st.chat_input(prompt_placeholder)
