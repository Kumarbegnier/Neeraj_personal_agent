from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import streamlit as st

from src.schemas.catalog import AgentCatalog, AuditEvent, ToolCatalog
from src.schemas.platform import HealthResponse, PlanResponse

from frontend.utils.state import current_memory
from frontend.view_models import (
    agent_rows,
    architecture_rows,
    audit_rows,
    conversation_rows,
    memory_record_rows,
    plan_step_rows,
    state_transition_rows,
    summarize_memory,
    task_graph_rows,
    tool_descriptor_rows,
    tool_result_rows,
    trace_rows,
)
from src.runtime.models import InteractionResponse, MemorySnapshot, SessionState, ToolResult
from src.runtime.workflow import StageDescriptor


def render_empty_state(title: str, detail: str) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.caption(detail)


def _render_dataframe_or_caption(
    rows: Sequence[dict[str, Any]],
    empty_message: str,
) -> None:
    if rows:
        st.dataframe(list(rows), use_container_width=True, hide_index=True)
    else:
        st.caption(empty_message)


def render_plan_panel(
    interaction: InteractionResponse | None,
    plan_preview: PlanResponse | None,
) -> None:
    plan = interaction.plan if interaction is not None else plan_preview.plan if plan_preview is not None else None
    task_graph = (
        interaction.task_graph
        if interaction is not None
        else plan_preview.task_graph
        if plan_preview is not None
        else None
    )

    st.subheader("Planner / Task Breakdown")
    if plan is None:
        render_empty_state(
            "No plan available.",
            "Use the planner preview or run a chat turn to populate the breakdown panel.",
        )
        return

    overview = st.columns(3)
    overview[0].metric("Strategy", plan.decomposition_strategy)
    overview[1].metric("Completion", plan.completion_state)
    overview[2].metric("Step count", len(plan.steps))
    st.caption(plan.reasoning)

    with st.container(border=True):
        st.markdown("**Execution Plan**")
        _render_dataframe_or_caption(
            plan_step_rows(plan),
            "No execution steps were generated for this plan preview.",
        )

    with st.container(border=True):
        st.markdown("**Task Graph**")
        _render_dataframe_or_caption(
            task_graph_rows(task_graph),
            "The task graph has not been populated yet.",
        )


def render_memory_panel(
    memory: MemorySnapshot | None,
    session_snapshot: SessionState | None,
) -> None:
    st.subheader("Memory / Context")
    if memory is None:
        render_empty_state(
            "No memory snapshot available.",
            "Run a request or refresh the backend session snapshot to inspect working memory.",
        )
        return

    metrics = st.columns(4)
    metrics[0].metric("Retrieved", len(memory.retrieved))
    metrics[1].metric("Semantic", len(memory.semantic))
    metrics[2].metric("Open loops", len(memory.open_loops))
    metrics[3].metric("Goals", len(memory.goal_stack))

    st.caption(summarize_memory(memory))

    tabs = st.tabs(["Working Memory", "Retrieved", "History"])
    with tabs[0]:
        with st.container(border=True):
            st.json(memory.working_memory.model_dump(), expanded=True)
    with tabs[1]:
        with st.container(border=True):
            _render_dataframe_or_caption(
                memory_record_rows(memory.retrieved),
                "No retrieved memory records are available for this session yet.",
            )
    with tabs[2]:
        _render_dataframe_or_caption(
            conversation_rows(session_snapshot.history) if session_snapshot is not None else [],
            "No durable conversation history has been loaded for this session yet.",
        )


def render_memory_collections(memory: MemorySnapshot | None) -> None:
    if memory is None:
        return

    semantic, vector = st.columns(2, gap="large")
    with semantic:
        with st.container(border=True):
            st.subheader("Semantic Memory")
            _render_dataframe_or_caption(
                memory_record_rows(memory.semantic),
                "No semantic memory entries are available.",
            )
    with vector:
        with st.container(border=True):
            st.subheader("Vector Memory")
            _render_dataframe_or_caption(
                memory_record_rows(memory.vector),
                "No vector memory entries are available.",
            )


def render_execution_panel(interaction: InteractionResponse | None) -> None:
    st.subheader("Agent Status / Execution")
    if interaction is None:
        render_empty_state(
            "No execution state yet.",
            "This panel will populate after the first completed backend interaction.",
        )
        return

    metrics = st.columns(5)
    metrics[0].metric("Agent", interaction.assigned_agent)
    metrics[1].metric("Loop count", interaction.loop_count)
    metrics[2].metric("Termination", interaction.termination_reason or "unknown")
    metrics[3].metric("Risk", interaction.safety.risk_level)
    metrics[4].metric("Approval", "Required" if interaction.safety.permission.requires_confirmation else "Clear")

    with st.status(
        label=f"Execution summary | verification={interaction.verification.status} | reflection={interaction.reflection.status}",
        state="complete" if interaction.termination_reason == "goal_achieved" else "running",
        expanded=True,
    ):
        st.write(interaction.response)
        st.caption(interaction.confirmation)

    st.markdown("**Tool Execution Status**")
    render_tool_result_cards(interaction.tool_results)


def render_tool_result_cards(results: Sequence[ToolResult]) -> None:
    if not results:
        st.caption("No tool results have been recorded yet.")
        return

    columns = st.columns(2)
    for index, result in enumerate(results):
        column = columns[index % 2]
        with column:
            with st.container(border=True):
                st.markdown(f"**{result.tool_name}**")
                metrics = st.columns(3)
                metrics[0].metric("Status", result.status)
                metrics[1].metric("Risk", result.risk_level)
                metrics[2].metric("Blocked", result.blocked_reason or "No")
                if result.evidence:
                    st.caption("Evidence")
                    for evidence in result.evidence[:3]:
                        st.write(f"- {evidence}")
                with st.expander("Structured output", expanded=False):
                    st.json(result.output, expanded=False)


def render_logs_panel(
    interaction: InteractionResponse | None,
    activity_log: Sequence[dict[str, str]],
    audit_events: Sequence[AuditEvent],
) -> None:
    st.subheader("Logs / Audit")
    if interaction is None and not activity_log and not audit_events:
        render_empty_state(
            "No logs yet.",
            "Run a prompt to capture trace events, tool status, and frontend activity.",
        )
        return

    tabs = st.tabs(["Runtime Trace", "Tool Logs", "State Transitions", "Audit Trail", "Frontend Activity"])
    with tabs[0]:
        if interaction is not None:
            _render_dataframe_or_caption(
                trace_rows(interaction.trace),
                "No runtime trace is available for the latest interaction.",
            )
        else:
            st.caption("No runtime trace available.")
    with tabs[1]:
        if interaction is not None:
            _render_dataframe_or_caption(
                tool_result_rows(interaction.tool_results),
                "No tool events were recorded for the latest interaction.",
            )
        else:
            st.caption("No tool events available.")
    with tabs[2]:
        if interaction is not None:
            _render_dataframe_or_caption(
                state_transition_rows(interaction.state_transitions),
                "No state transitions were recorded for the latest interaction.",
            )
        else:
            st.caption("No state transitions available.")
    with tabs[3]:
        _render_dataframe_or_caption(
            audit_rows(audit_events),
            "No backend audit events have been recorded yet.",
        )
    with tabs[4]:
        _render_dataframe_or_caption(
            list(activity_log),
            "No frontend activity has been recorded yet.",
        )


def render_runtime_catalog_panel(
    agent_catalog: AgentCatalog | None,
    tool_catalog: ToolCatalog | None,
) -> None:
    st.subheader("Specialists / Registry")
    if agent_catalog is None and tool_catalog is None:
        render_empty_state(
            "No runtime catalogs loaded.",
            "Refresh the runtime catalogs to inspect specialists and shared tools.",
        )
        return

    metrics = st.columns(2)
    metrics[0].metric("Agents", len(agent_catalog.agents) if agent_catalog is not None else 0)
    metrics[1].metric("Tools", len(tool_catalog.tools) if tool_catalog is not None else 0)

    tabs = st.tabs(["Agents", "Tools"])
    with tabs[0]:
        if agent_catalog is not None:
            _render_dataframe_or_caption(
                agent_rows(agent_catalog.agents),
                "No agent catalog is available from the backend.",
            )
        else:
            st.caption("No agent catalog is available from the backend.")
    with tabs[1]:
        if tool_catalog is not None:
            _render_dataframe_or_caption(
                tool_descriptor_rows(tool_catalog.tools),
                "No tool catalog is available from the backend.",
            )
        else:
            st.caption("No tool catalog is available from the backend.")


def render_agent_review_panels(
    interaction: InteractionResponse | None,
    plan_preview: PlanResponse | None,
) -> None:
    if interaction is None and plan_preview is None:
        render_empty_state(
            "No agent review data available.",
            "Preview a plan or complete a chat turn to inspect control, skills, and safety state.",
        )
        return

    control = interaction.control if interaction is not None else plan_preview.control
    skills = interaction.skills if interaction is not None else plan_preview.skills

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Control Decision")
            st.json(control.model_dump(), expanded=True)
    with right:
        with st.container(border=True):
            st.subheader("Skills")
            if skills:
                for skill in skills:
                    st.markdown(f"**{skill.name}**")
                    st.caption(skill.description)
            else:
                st.caption("No specialist skills were attached to the latest run.")

    if interaction is not None:
        review_tabs = st.tabs(["Verification", "Reflection", "Safety"])
        with review_tabs[0]:
            st.json(interaction.verification.model_dump(), expanded=True)
        with review_tabs[1]:
            st.json(interaction.reflection.model_dump(), expanded=True)
        with review_tabs[2]:
            st.json(interaction.safety.model_dump(), expanded=True)


def render_home_dashboard(
    state: MutableMapping[str, Any],
    health: HealthResponse | None,
    architecture: Sequence[StageDescriptor],
) -> None:
    current_interaction = state.get("last_interaction")
    metrics = st.columns(5)
    metrics[0].metric("Backend", health.status if health is not None else "unknown")
    metrics[1].metric("Agent", str(state["selected_agent"]))
    metrics[2].metric("Risk", str(state["risk_level"]).upper())
    metrics[3].metric("Messages", len(state["messages"]))
    metrics[4].metric("Approval", "Required" if state["approval_required"] else "Clear")

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("System Snapshot")
            if health is None:
                st.caption("Backend health has not been loaded yet.")
            else:
                st.json(health.model_dump(), expanded=True)

        with st.container(border=True):
            st.subheader("Architecture")
            _render_dataframe_or_caption(
                architecture_rows(architecture),
                "Architecture stages will appear after the first successful backend refresh.",
            )

    with right:
        with st.container(border=True):
            st.subheader("Latest Session Context")
            st.caption(summarize_memory(current_memory(state)))
            if current_interaction is not None:
                st.write(current_interaction.response)
            else:
                st.caption("No interaction has been completed yet.")

        with st.container(border=True):
            st.subheader("Conversation Snapshot")
            if state["messages"]:
                for entry in list(state["messages"])[-4:]:
                    st.caption(f"{entry.role} | {entry.timestamp}")
                    st.write(entry.content[:200])
            else:
                st.caption("The conversation transcript is currently empty.")
