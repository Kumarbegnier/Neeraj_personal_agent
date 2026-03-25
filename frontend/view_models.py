from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from typing import Any

from src.schemas.catalog import AgentDescriptor, AuditEvent, ToolDescriptor
from src.runtime.models import (
    AutonomyMetrics,
    ConversationTurn,
    ExecutionPlan,
    MemoryRecord,
    MemorySnapshot,
    RuntimeTrace,
    StateTransition,
    StepTrace,
    TaskGraph,
    ToolResult,
    TraceEvent,
)
from src.runtime.workflow import StageDescriptor


def humanize_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title() if value else "Unknown"


def selected_agent_label(agent_key: str, agents: Sequence[AgentDescriptor] | None) -> str:
    if not agent_key:
        return "Awaiting routing"

    for agent in agents or []:
        if agent.key == agent_key:
            return agent.display_name
    return humanize_label(agent_key)


def streamlit_status_state(status: str) -> Literal["running", "complete", "error"]:
    normalized = status.strip().lower()
    if normalized in {"complete", "completed", "success", "verified", "ready", "clear"}:
        return "complete"
    if normalized in {"blocked", "failed", "error", "rejected"}:
        return "error"
    return "running"


def compact_text(value: str, *, limit: int = 220) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def format_ratio(value: float) -> str:
    return f"{value:.0%}"


def summarize_memory(memory: MemorySnapshot | None) -> str:
    if memory is None:
        return "No memory snapshot has been loaded yet."

    parts = [memory.summary]
    if memory.goal_stack:
        parts.append(f"Goals: {', '.join(memory.goal_stack[:3])}")
    if memory.open_loops:
        parts.append(f"Open loops: {', '.join(memory.open_loops[:2])}")
    return " | ".join(part for part in parts if part).strip()[:280]


def plan_step_rows(plan: ExecutionPlan | None) -> list[dict[str, Any]]:
    if plan is None:
        return []
    return [
        {
            "Step": step.name,
            "Owner": step.owner,
            "Status": step.status,
            "Type": step.step_type,
            "Risk": step.risk_level,
            "Depends On": ", ".join(step.depends_on) or "None",
            "Tools": ", ".join(step.requires_tools) or "None",
        }
        for step in plan.steps
    ]


def task_graph_rows(task_graph: TaskGraph | None) -> list[dict[str, Any]]:
    if task_graph is None:
        return []
    return [
        {
            "Node": node.name,
            "Owner": node.owner,
            "Status": node.status,
            "Depends On": ", ".join(node.depends_on) or "None",
            "Verification": "Yes" if node.verification_required else "No",
        }
        for node in task_graph.nodes
    ]


def memory_record_rows(records: Sequence[MemoryRecord]) -> list[dict[str, Any]]:
    return [
        {
            "Type": record.memory_type,
            "Source": record.source,
            "Salience": record.salience,
            "Score": record.score,
            "Tags": ", ".join(record.tags) or "None",
            "Content": record.content,
        }
        for record in records
    ]


def conversation_rows(history: Sequence[ConversationTurn]) -> list[dict[str, Any]]:
    return [
        {
            "Role": turn.role,
            "Timestamp": turn.timestamp.isoformat(),
            "Content": turn.content,
        }
        for turn in history
    ]


def tool_result_rows(results: Sequence[ToolResult]) -> list[dict[str, Any]]:
    return [
        {
            "Tool": result.tool_name,
            "Status": result.status,
            "Risk": result.risk_level,
            "Blocked Reason": result.blocked_reason or "None",
            "Evidence": " | ".join(result.evidence[:2]) or "No evidence captured.",
        }
        for result in results
    ]


def trace_rows(events: Sequence[TraceEvent]) -> list[dict[str, Any]]:
    return [
        {
            "Stage": event.stage,
            "Detail": event.detail,
            "Payload Keys": ", ".join(event.payload.keys()) or "None",
        }
        for event in events
    ]


def step_trace_rows(steps: Sequence[StepTrace]) -> list[dict[str, Any]]:
    return [
        {
            "Step": step.step_index,
            "Agent": humanize_label(step.agent_name),
            "Phase": humanize_label(step.phase),
            "Status": humanize_label(step.status),
            "Autonomous": "Yes" if step.autonomous else "No",
            "Approvals": step.approvals_requested,
            "Retry": "Yes" if step.retry_triggered else "No",
            "Recovered": "Yes" if step.recovered_after_failure else "No",
            "Tools": ", ".join(step.selected_tools) or "None",
            "Summary": compact_text(step.summary, limit=120),
        }
        for step in steps
    ]


def runtime_trace_rows(traces: Sequence[RuntimeTrace]) -> list[dict[str, Any]]:
    return [
        {
            "Recorded At": trace.recorded_at.isoformat(),
            "Request": trace.request_id or "Unknown",
            "Agent": humanize_label(trace.assigned_agent),
            "Architecture": humanize_label(trace.architecture_mode),
            "Termination": humanize_label(trace.termination_reason),
            "Steps": trace.autonomy_metrics.total_steps,
            "Autonomous": trace.autonomy_metrics.autonomous_steps_count,
            "Human Ratio": format_ratio(trace.autonomy_metrics.human_intervention_ratio),
            "Summary": compact_text(trace.summary, limit=120),
        }
        for trace in traces
    ]


def autonomy_metrics_rows(metrics: AutonomyMetrics | None) -> list[dict[str, Any]]:
    if metrics is None:
        return []
    return [
        {"Metric": "Autonomous steps", "Value": metrics.autonomous_steps_count},
        {"Metric": "Approvals requested", "Value": metrics.approvals_requested},
        {"Metric": "Retries used", "Value": metrics.retries_used},
        {"Metric": "Recoveries after failure", "Value": metrics.recovery_count_after_failure},
        {"Metric": "Human intervention ratio", "Value": format_ratio(metrics.human_intervention_ratio)},
        {"Metric": "Irreversible actions attempted", "Value": metrics.irreversible_actions_attempted},
    ]


def state_transition_rows(transitions: Sequence[StateTransition]) -> list[dict[str, Any]]:
    return [
        {
            "Step": transition.step_index,
            "Route": transition.selected_route,
            "Retry": "Yes" if transition.retry_recommended else "No",
            "Replan": "Yes" if transition.replan_required else "No",
            "Ready": "Yes" if transition.ready_for_response else "No",
            "Signal": transition.termination_signal,
        }
        for transition in transitions
    ]


def architecture_rows(stages: Sequence[StageDescriptor]) -> list[dict[str, Any]]:
    return [
        {
            "Stage": stage.name,
            "Component": stage.component,
            "Description": stage.description,
        }
        for stage in stages
    ]


def agent_rows(agents: Sequence[AgentDescriptor]) -> list[dict[str, Any]]:
    return [
        {
            "Key": agent.key,
            "Display Name": agent.display_name,
            "Role": agent.role,
            "Default Tools": ", ".join(agent.default_tools) or "None",
            "Description": agent.description,
        }
        for agent in agents
    ]


def tool_descriptor_rows(tools: Sequence[ToolDescriptor]) -> list[dict[str, Any]]:
    return [
        {
            "Name": tool.name,
            "Category": tool.category,
            "Risk": tool.risk_level,
            "Side Effect": tool.side_effect,
            "Description": tool.description,
        }
        for tool in tools
    ]


def audit_rows(events: Sequence[AuditEvent]) -> list[dict[str, Any]]:
    return [
        {
            "Timestamp": event.recorded_at.isoformat(),
            "Event": event.event,
            "Payload Keys": ", ".join(event.payload.keys()) or "None",
        }
        for event in events
    ]
