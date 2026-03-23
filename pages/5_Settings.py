from __future__ import annotations

import streamlit as st

from frontend.bootstrap import bootstrap_page
from frontend.components.chat_view import render_runtime_notices
from frontend.controller import clear_workspace, refresh_health, start_new_session


service, config = bootstrap_page(
    "Settings",
    "Backend connection, local session controls, and frontend execution preferences.",
)

render_runtime_notices(st.session_state)

st.write(
    "Adjust the frontend connection and local runtime preferences here. These controls update Streamlit "
    "session state, which the UI uses for model choice, session identity, and approval behavior."
)

with st.form("frontend_settings"):
    model_options = list(config.model_options)
    selected_model = str(st.session_state["selected_model"])
    selected_index = model_options.index(selected_model) if selected_model in model_options else 0
    backend_url = st.text_input("Backend URL", value=str(st.session_state["backend_url"]))
    user_id = st.text_input("User ID", value=str(st.session_state["user_id"]))
    session_id = st.text_input("Session ID", value=str(st.session_state["session_id"]))
    model_selection = st.selectbox(
        "Preferred model",
        options=model_options,
        index=selected_index,
    )
    approval_toggle = st.toggle(
        "Auto-approve gated actions",
        value=bool(st.session_state["approval_toggle"]),
    )
    submitted = st.form_submit_button("Apply settings", use_container_width=True)

if submitted:
    st.session_state["backend_url"] = backend_url
    st.session_state["user_id"] = user_id
    st.session_state["session_id"] = session_id
    st.session_state["selected_model"] = model_selection
    st.session_state["approval_toggle"] = approval_toggle
    st.rerun()

actions = st.columns(3)
if actions[0].button("Refresh backend health", use_container_width=True):
    with st.spinner("Refreshing backend health..."):
        refresh_health(st.session_state, service)
    st.rerun()
if actions[1].button("Start new session", use_container_width=True):
    start_new_session(st.session_state)
    st.rerun()
if actions[2].button("Clear workspace", use_container_width=True):
    clear_workspace(st.session_state)
    st.rerun()

with st.container(border=True):
    st.subheader("Current frontend configuration")
    st.caption(
        "Model selection is currently stored in frontend metadata and request preferences so the UI remains "
        "aligned with future backend model-routing upgrades."
    )
    st.json(
        {
            "backend_url": st.session_state["backend_url"],
            "user_id": st.session_state["user_id"],
            "session_id": st.session_state["session_id"],
            "selected_model": st.session_state["selected_model"],
            "approval_toggle": st.session_state["approval_toggle"],
        },
        expanded=True,
    )

if st.session_state.get("last_health") is not None:
    with st.container(border=True):
        st.subheader("Latest backend health")
        st.json(st.session_state["last_health"].model_dump(), expanded=True)
