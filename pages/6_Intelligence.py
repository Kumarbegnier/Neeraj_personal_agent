from __future__ import annotations

import streamlit as st

from frontend.bootstrap import bootstrap_page
from frontend.components.chat_view import render_runtime_notices
from frontend.components.intelligence_panels import render_runtime_intelligence_dashboard
from frontend.controller import refresh_runtime_intelligence
from frontend.utils.state import current_memory


service, _config = bootstrap_page(
    "Runtime Intelligence",
    "Research-lab view of routing, architecture, evaluation history, and execution intelligence.",
)

render_runtime_notices(st.session_state)

if (
    st.session_state.get("last_health") is None
    or st.session_state.get("last_session_snapshot") is None
    or not st.session_state.get("evaluation_winners")
):
    with st.spinner("Loading runtime intelligence..."):
        refresh_runtime_intelligence(st.session_state, service)

st.write(
    "Use this page as the runtime intelligence bench: it surfaces the current provider path, the selected "
    "execution architecture, tool progression, reflection state, autonomy posture, routing winners by task "
    "family, and the evidence currently sitting in retrieved memory."
)

controls = st.columns([0.45, 0.25, 0.3], gap="large")
winners_limit = controls[0].selectbox(
    "Winner depth",
    options=[8, 12, 20],
    index=1,
)
trace_limit = controls[1].selectbox(
    "Trace depth",
    options=[8, 12, 25],
    index=2,
)
if controls[2].button("Refresh intelligence", use_container_width=True):
    with st.spinner("Refreshing runtime intelligence..."):
        refresh_runtime_intelligence(
            st.session_state,
            service,
            winners_limit=int(winners_limit),
            trace_limit=int(trace_limit),
        )
    st.rerun()

render_runtime_intelligence_dashboard(
    st.session_state.get("last_interaction"),
    st.session_state.get("last_health"),
    current_memory(st.session_state),
    st.session_state.get("evaluation_winners", []),
    st.session_state.get("runtime_traces", []),
)
