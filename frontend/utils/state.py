from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, MutableMapping
from uuid import uuid4

from src.schemas.catalog import AgentCatalog, AuditEvent, ToolCatalog
from src.schemas.platform import HealthResponse, PlanResponse

from frontend.config import FrontendConfig
from frontend.view_models import summarize_memory
from src.runtime.models import InteractionResponse, MemorySnapshot, SessionState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_session_id() -> str:
    return f"session-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class TranscriptEntry:
    role: str
    content: str
    caption: str = ""
    timestamp: str = field(default_factory=_utc_now)


def ensure_session_state(state: MutableMapping[str, Any], config: FrontendConfig) -> None:
    defaults = {
        "backend_url": config.backend_url,
        "user_id": "streamlit-user",
        "session_id": _new_session_id(),
        "selected_model": config.model_options[0] if config.model_options else "backend-default",
        "selected_agent": "Awaiting first routing decision",
        "agent_status": "idle",
        "risk_level": "unknown",
        "approval_toggle": False,
        "approval_required": False,
        "memory_summary": "No memory has been collected yet.",
        "messages": [],
        "activity_log": [],
        "architecture": [],
        "last_prompt": "",
        "pending_prompt": None,
        "last_error": "",
        "last_health": None,
        "last_plan": None,
        "last_interaction": None,
        "last_session_snapshot": None,
        "agent_catalog": None,
        "tool_catalog": None,
        "audit_events": [],
    }
    for key, value in defaults.items():
        state.setdefault(key, value)


def append_message(
    state: MutableMapping[str, Any],
    role: str,
    content: str,
    *,
    caption: str = "",
) -> None:
    state["messages"].append(TranscriptEntry(role=role, content=content, caption=caption))


def record_activity(
    state: MutableMapping[str, Any],
    event: str,
    detail: str,
) -> None:
    state["activity_log"].append(
        {
            "Timestamp": _utc_now(),
            "Event": event,
            "Detail": detail,
        }
    )
    state["activity_log"] = state["activity_log"][-60:]


def clear_error(state: MutableMapping[str, Any]) -> None:
    state["last_error"] = ""


def set_error(state: MutableMapping[str, Any], error_message: str) -> None:
    state["last_error"] = error_message


def sync_health(state: MutableMapping[str, Any], health: HealthResponse) -> None:
    state["last_health"] = health


def sync_plan(state: MutableMapping[str, Any], plan: PlanResponse) -> None:
    state["last_plan"] = plan
    state["selected_agent"] = plan.assigned_agent
    state["agent_status"] = plan.plan.completion_state
    state["risk_level"] = plan.control.risk_level
    state["approval_required"] = plan.permission.requires_confirmation
    state["memory_summary"] = summarize_memory(plan.memory)


def sync_interaction(
    state: MutableMapping[str, Any],
    interaction: InteractionResponse,
) -> None:
    state["last_interaction"] = interaction
    state["selected_agent"] = interaction.assigned_agent
    state["agent_status"] = interaction.termination_reason or "completed"
    state["risk_level"] = interaction.safety.risk_level
    state["approval_required"] = interaction.safety.permission.requires_confirmation
    state["memory_summary"] = summarize_memory(interaction.memory)
    state["pending_prompt"] = state["last_prompt"] if interaction.safety.permission.requires_confirmation else None


def sync_session_snapshot(
    state: MutableMapping[str, Any],
    session_snapshot: SessionState,
) -> None:
    state["last_session_snapshot"] = session_snapshot
    state["memory_summary"] = summarize_memory(session_snapshot.memory)


def sync_agent_catalog(
    state: MutableMapping[str, Any],
    catalog: AgentCatalog,
) -> None:
    state["agent_catalog"] = catalog


def sync_tool_catalog(
    state: MutableMapping[str, Any],
    catalog: ToolCatalog,
) -> None:
    state["tool_catalog"] = catalog


def sync_audit_events(
    state: MutableMapping[str, Any],
    events: list[AuditEvent],
) -> None:
    state["audit_events"] = events


def current_memory(state: MutableMapping[str, Any]) -> MemorySnapshot | None:
    interaction = state.get("last_interaction")
    if interaction is not None:
        return interaction.memory
    snapshot = state.get("last_session_snapshot")
    if snapshot is not None:
        return snapshot.memory
    plan = state.get("last_plan")
    if plan is not None:
        return plan.memory
    return None


def reset_workspace(state: MutableMapping[str, Any], *, new_session: bool = False) -> None:
    state["messages"] = []
    state["activity_log"] = []
    state["last_prompt"] = ""
    state["pending_prompt"] = None
    state["last_error"] = ""
    state["last_plan"] = None
    state["last_interaction"] = None
    state["last_session_snapshot"] = None
    state["audit_events"] = []
    state["selected_agent"] = "Awaiting first routing decision"
    state["agent_status"] = "idle"
    state["risk_level"] = "unknown"
    state["approval_required"] = False
    state["memory_summary"] = "No memory has been collected yet."
    if new_session:
        state["session_id"] = _new_session_id()
