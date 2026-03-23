from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import streamlit as st

from src.schemas.catalog import AgentCatalog, AuditEvent, ToolCatalog
from src.schemas.platform import HealthResponse, PlanResponse

from frontend.components.primitives import (
    MetricSpec,
    render_bullet_card,
    render_dataframe_card,
    render_dataframe_or_caption,
    render_empty_state,
    render_json_card,
    render_metric_strip,
    render_section_intro,
    render_text_card,
)
from frontend.utils.state import current_memory
from frontend.view_models import (
    agent_rows,
    architecture_rows,
    audit_rows,
    compact_text,
    conversation_rows,
    humanize_label,
    memory_record_rows,
    plan_step_rows,
    selected_agent_label,
    state_transition_rows,
    streamlit_status_state,
    summarize_memory,
    task_graph_rows,
    tool_descriptor_rows,
    tool_result_rows,
    trace_rows,
)
from src.runtime.models import InteractionResponse, MemorySnapshot, SessionState, ToolResult
from src.runtime.workflow import StageDescriptor


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

    render_section_intro(
        "Planner / Task Breakdown",
        "Inspect the current decomposition strategy, task graph, and verification shape.",
    )
    if plan is None:
        render_empty_state(
            "No plan available.",
            "Use the planner preview or run a chat turn to populate the breakdown panel.",
        )
        return

    render_metric_strip(
        [
            MetricSpec("Strategy", humanize_label(plan.decomposition_strategy or "not_set")),
            MetricSpec("Completion", humanize_label(plan.completion_state)),
            MetricSpec("Plan steps", len(plan.steps)),
            MetricSpec("Task nodes", len(task_graph.nodes) if task_graph is not None else 0),
        ]
    )

    tabs = st.tabs(["Plan", "Task Graph", "Reasoning", "Quality Controls"])
    with tabs[0]:
        render_dataframe_card(
            "Execution Plan",
            plan_step_rows(plan),
            empty_message="No execution steps were generated for this plan.",
            caption=plan.objective,
        )
    with tabs[1]:
        render_dataframe_card(
            "Task Graph",
            task_graph_rows(task_graph),
            empty_message="The task graph has not been populated yet.",
            caption=f"Graph state: {humanize_label(task_graph.state) if task_graph is not None else 'Unavailable'}",
        )
    with tabs[2]:
        render_text_card(
            "Planner Rationale",
            plan.reasoning,
            empty_message="The planner did not return reasoning text.",
        )
        support_left, support_right = st.columns(2, gap="large")
        with support_left:
            render_bullet_card(
                "Assumptions",
                plan.assumptions,
                empty_message="No assumptions were recorded for the current plan.",
            )
        with support_right:
            render_bullet_card(
                "Constraints",
                plan.constraints,
                empty_message="No constraints were recorded for the current plan.",
            )
    with tabs[3]:
        quality_left, quality_right = st.columns(2, gap="large")
        with quality_left:
            render_bullet_card(
                "Success Criteria",
                plan.success_criteria,
                empty_message="No success criteria were attached to the current plan.",
            )
            render_bullet_card(
                "Verification Focus",
                plan.verification_focus,
                empty_message="No verification focus areas were attached to the current plan.",
            )
        with quality_right:
            render_bullet_card(
                "Failure Modes",
                plan.failure_modes,
                empty_message="No failure modes were attached to the current plan.",
            )


def render_memory_panel(
    memory: MemorySnapshot | None,
    session_snapshot: SessionState | None,
) -> None:
    render_section_intro(
        "Memory / Context",
        "Review working memory, retrieved context, active goals, and session-level history.",
    )
    if memory is None:
        render_empty_state(
            "No memory snapshot available.",
            "Run a request or refresh the backend session snapshot to inspect working memory.",
        )
        return

    render_metric_strip(
        [
            MetricSpec("Retrieved", len(memory.retrieved)),
            MetricSpec("Semantic", len(memory.semantic)),
            MetricSpec("Open loops", len(memory.open_loops)),
            MetricSpec("Goals", len(memory.goal_stack)),
        ]
    )

    tabs = st.tabs(["Summary", "Working Memory", "Retrieved", "History"])
    with tabs[0]:
        render_text_card(
            "Memory Summary",
            summarize_memory(memory),
            empty_message="No memory summary is available.",
        )
        left, right = st.columns(2, gap="large")
        with left:
            render_bullet_card(
                "Goal Stack",
                memory.goal_stack,
                empty_message="No active goals are recorded in memory.",
            )
            render_bullet_card(
                "Open Loops",
                memory.open_loops,
                empty_message="No open loops are currently tracked.",
            )
        with right:
            render_bullet_card(
                "Open Questions",
                memory.working_memory.open_questions,
                empty_message="No open questions are tracked in working memory.",
            )
            render_bullet_card(
                "Constraints",
                memory.working_memory.constraints,
                empty_message="No working constraints are currently stored.",
            )
    with tabs[1]:
        render_json_card(
            "Working Memory Object",
            memory.working_memory.model_dump(),
            empty_message="Working memory is not available.",
            caption="This is the live distilled memory object returned by the backend.",
        )
    with tabs[2]:
        render_dataframe_card(
            "Retrieved Memory Records",
            memory_record_rows(memory.retrieved),
            empty_message="No retrieved memory records are available for this session yet.",
        )
    with tabs[3]:
        render_dataframe_card(
            "Session Conversation History",
            conversation_rows(session_snapshot.history) if session_snapshot is not None else [],
            empty_message="No durable conversation history has been loaded for this session yet.",
        )
        if session_snapshot is not None and session_snapshot.preferences:
            render_json_card(
                "Session Preferences",
                session_snapshot.preferences,
                empty_message="No session preferences are stored yet.",
            )


def render_memory_collections(memory: MemorySnapshot | None) -> None:
    if memory is None:
        return

    semantic, vector = st.columns(2, gap="large")
    with semantic:
        render_dataframe_card(
            "Semantic Memory",
            memory_record_rows(memory.semantic),
            empty_message="No semantic memory entries are available.",
        )
    with vector:
        render_dataframe_card(
            "Vector Memory",
            memory_record_rows(memory.vector),
            empty_message="No vector memory entries are available.",
        )


def render_execution_panel(interaction: InteractionResponse | None) -> None:
    render_section_intro(
        "Agent Status / Execution",
        "Monitor routed execution, verification results, safety posture, and tool progress.",
    )
    if interaction is None:
        render_empty_state(
            "No execution state yet.",
            "This panel will populate after the first completed backend interaction.",
        )
        return

    render_metric_strip(
        [
            MetricSpec("Agent", humanize_label(interaction.assigned_agent)),
            MetricSpec("Loop count", interaction.loop_count),
            MetricSpec("Termination", humanize_label(interaction.termination_reason or "unknown")),
            MetricSpec("Risk", humanize_label(interaction.safety.risk_level)),
            MetricSpec(
                "Approval",
                "Required" if interaction.safety.permission.requires_confirmation else "Clear",
            ),
        ]
    )

    with st.status(
        label=(
            "Execution summary | "
            f"verification={humanize_label(interaction.verification.status)} | "
            f"reflection={humanize_label(interaction.reflection.status)}"
        ),
        state=streamlit_status_state(interaction.verification.status),
        expanded=True,
    ):
        st.write(interaction.response)
        st.caption(interaction.confirmation)
        if interaction.safety.notes:
            st.write("Safety notes")
            for note in interaction.safety.notes[:4]:
                st.write(f"- {note}")

    tabs = st.tabs(["Tool Status", "Verification", "Reflection", "Safety"])
    with tabs[0]:
        render_tool_result_cards(interaction.tool_results)
    with tabs[1]:
        left, right = st.columns(2, gap="large")
        with left:
            render_text_card(
                "Verification Summary",
                interaction.verification.summary,
                empty_message="No verification summary is available.",
            )
            render_bullet_card(
                "Verified Claims",
                interaction.verification.verified_claims,
                empty_message="No claims were marked as verified.",
            )
        with right:
            render_bullet_card(
                "Verification Gaps",
                interaction.verification.gaps,
                empty_message="No verification gaps were reported.",
            )
            render_bullet_card(
                "Unverified Claims",
                interaction.verification.unverified_claims,
                empty_message="No unverified claims were reported.",
            )
    with tabs[2]:
        left, right = st.columns(2, gap="large")
        with left:
            render_text_card(
                "Reflection Summary",
                interaction.reflection.summary,
                empty_message="No reflection summary is available.",
            )
            render_bullet_card(
                "Repairs",
                interaction.reflection.repairs,
                empty_message="No runtime repairs were recorded.",
            )
        with right:
            render_bullet_card(
                "Issues",
                interaction.reflection.issues,
                empty_message="No reflection issues were reported.",
            )
            render_bullet_card(
                "Next Actions",
                interaction.reflection.next_actions,
                empty_message="No follow-up actions were recorded.",
            )
    with tabs[3]:
        render_json_card(
            "Safety Report",
            interaction.safety.model_dump(),
            empty_message="No safety report is available.",
        )


def render_tool_result_cards(results: Sequence[ToolResult]) -> None:
    if not results:
        st.caption("No tool results have been recorded yet.")
        return

    columns = st.columns(2)
    for index, result in enumerate(results):
        column = columns[index % 2]
        with column:
            with st.status(
                label=f"{result.tool_name} | {humanize_label(result.status)}",
                state=streamlit_status_state(result.status),
                expanded=False,
            ):
                metrics = st.columns(3)
                metrics[0].metric("Status", humanize_label(result.status))
                metrics[1].metric("Risk", humanize_label(result.risk_level))
                metrics[2].metric("Blocked", "Yes" if result.blocked_reason else "No")
                if result.blocked_reason:
                    st.caption(f"Blocked reason: {result.blocked_reason}")
                if result.evidence:
                    st.caption("Evidence")
                    for evidence in result.evidence[:3]:
                        st.write(f"- {evidence}")
                else:
                    st.caption("No evidence was captured for this tool call.")
                with st.expander("Structured output", expanded=False):
                    st.json(result.output, expanded=False)


def render_logs_panel(
    interaction: InteractionResponse | None,
    activity_log: Sequence[dict[str, str]],
    audit_events: Sequence[AuditEvent],
) -> None:
    render_section_intro(
        "Logs / Audit",
        "Inspect traces, tool events, state transitions, backend audit records, and frontend activity.",
    )
    if interaction is None and not activity_log and not audit_events:
        render_empty_state(
            "No logs yet.",
            "Run a prompt to capture trace events, tool status, and frontend activity.",
        )
        return

    render_metric_strip(
        [
            MetricSpec("Trace events", len(interaction.trace) if interaction is not None else 0),
            MetricSpec("Tool logs", len(interaction.tool_results) if interaction is not None else 0),
            MetricSpec("Transitions", len(interaction.state_transitions) if interaction is not None else 0),
            MetricSpec("Audit records", len(audit_events)),
        ]
    )

    tabs = st.tabs(["Runtime Trace", "Tool Logs", "State Transitions", "Audit Trail", "Frontend Activity"])
    with tabs[0]:
        if interaction is not None:
            render_dataframe_or_caption(
                trace_rows(interaction.trace),
                "No runtime trace is available for the latest interaction.",
            )
        else:
            st.caption("No runtime trace available.")
    with tabs[1]:
        if interaction is not None:
            render_dataframe_or_caption(
                tool_result_rows(interaction.tool_results),
                "No tool events were recorded for the latest interaction.",
            )
        else:
            st.caption("No tool events available.")
    with tabs[2]:
        if interaction is not None:
            render_dataframe_or_caption(
                state_transition_rows(interaction.state_transitions),
                "No state transitions were recorded for the latest interaction.",
            )
        else:
            st.caption("No state transitions available.")
    with tabs[3]:
        render_dataframe_or_caption(
            audit_rows(audit_events),
            "No backend audit events have been recorded yet.",
        )
    with tabs[4]:
        render_dataframe_or_caption(
            list(activity_log),
            "No frontend activity has been recorded yet.",
        )


def render_runtime_catalog_panel(
    agent_catalog: AgentCatalog | None,
    tool_catalog: ToolCatalog | None,
) -> None:
    render_section_intro(
        "Specialists / Registry",
        "Inspect the backend-exposed specialist catalog and shared tool registry.",
    )
    if agent_catalog is None and tool_catalog is None:
        render_empty_state(
            "No runtime catalogs loaded.",
            "Refresh the runtime catalogs to inspect specialists and shared tools.",
        )
        return

    render_metric_strip(
        [
            MetricSpec("Agents", len(agent_catalog.agents) if agent_catalog is not None else 0),
            MetricSpec("Tools", len(tool_catalog.tools) if tool_catalog is not None else 0),
        ]
    )

    tabs = st.tabs(["Agents", "Tools"])
    with tabs[0]:
        if agent_catalog is not None:
            render_dataframe_or_caption(
                agent_rows(agent_catalog.agents),
                "No agent catalog is available from the backend.",
            )
        else:
            st.caption("No agent catalog is available from the backend.")
    with tabs[1]:
        if tool_catalog is not None:
            render_dataframe_or_caption(
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
        render_json_card(
            "Control Decision",
            control.model_dump(),
            empty_message="No control decision is available.",
        )
    with right:
        render_bullet_card(
            "Specialist Skills",
            [f"{skill.name}: {compact_text(skill.description)}" for skill in skills],
            empty_message="No specialist skills were attached to the latest run.",
        )

    if interaction is not None:
        review_tabs = st.tabs(["Verification", "Reflection", "Safety"])
        with review_tabs[0]:
            render_json_card(
                "Verification Detail",
                interaction.verification.model_dump(),
                empty_message="No verification detail is available.",
            )
        with review_tabs[1]:
            render_json_card(
                "Reflection Detail",
                interaction.reflection.model_dump(),
                empty_message="No reflection detail is available.",
            )
        with review_tabs[2]:
            render_json_card(
                "Safety Detail",
                interaction.safety.model_dump(),
                empty_message="No safety detail is available.",
            )


def render_home_dashboard(
    state: MutableMapping[str, Any],
    health: HealthResponse | None,
    architecture: Sequence[StageDescriptor],
) -> None:
    current_interaction = state.get("last_interaction")
    agent_catalog = state.get("agent_catalog")
    render_metric_strip(
        [
            MetricSpec("Backend", humanize_label(health.status) if health is not None else "Unknown"),
            MetricSpec(
                "Agent",
                selected_agent_label(
                    str(state["selected_agent"]),
                    getattr(agent_catalog, "agents", None),
                ),
            ),
            MetricSpec("Risk", humanize_label(str(state["risk_level"]))),
            MetricSpec("Messages", len(state["messages"])),
            MetricSpec("Approval", "Required" if state["approval_required"] else "Clear"),
        ]
    )

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        render_json_card(
            "System Snapshot",
            health.model_dump() if health is not None else None,
            empty_message="Backend health has not been loaded yet.",
            caption="FastAPI health data, memory availability, and runtime integration status.",
        )
        render_dataframe_card(
            "Architecture",
            architecture_rows(architecture),
            empty_message="Architecture stages will appear after the first successful backend refresh.",
            caption="The backend exposes the active architecture stages to the UI for inspection.",
        )

    with right:
        render_text_card(
            "Latest Session Context",
            summarize_memory(current_memory(state)),
            empty_message="No memory summary is available yet.",
        )
        render_text_card(
            "Latest Response",
            current_interaction.response if current_interaction is not None else "",
            empty_message="No interaction has been completed yet.",
            caption="The most recent final response synthesized by the backend runtime.",
        )
        with st.container(border=True):
            st.markdown("**Conversation Snapshot**")
            if state["messages"]:
                for entry in list(state["messages"])[-4:]:
                    st.caption(f"{humanize_label(entry.role)} | {entry.timestamp}")
                    st.write(compact_text(entry.content, limit=180))
            else:
                st.caption("The conversation transcript is currently empty.")
