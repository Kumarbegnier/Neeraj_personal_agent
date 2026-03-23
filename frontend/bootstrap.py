from __future__ import annotations

import streamlit as st

from .components.sidebar import render_sidebar
from .config import FrontendConfig, get_frontend_config
from .controller import build_api_client
from .services.api_client import ApiClient
from .utils.state import ensure_session_state


def bootstrap_page(
    title: str,
    subtitle: str,
) -> tuple[ApiClient, FrontendConfig]:
    config = get_frontend_config()
    st.set_page_config(
        page_title=f"{config.app_title} | {title}",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_session_state(st.session_state, config)
    render_sidebar(st.session_state, config)
    st.title(title)
    st.caption(subtitle)
    return build_api_client(st.session_state, config), config
