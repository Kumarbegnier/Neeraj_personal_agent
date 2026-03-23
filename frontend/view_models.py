from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.schemas.catalog import AgentDescriptor, AuditEvent, ToolDescriptor
from src.runtime.models import (
    ConversationTurn,
    ExecutionPlan,
    MemoryRecord,
    MemorySnapshot,
    StateTransition,
    TaskGraph,
    ToolResult,
    TraceEvent,
)
from src.runtime.workflow import StageDescriptor


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
