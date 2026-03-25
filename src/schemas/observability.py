from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_trace_id() -> str:
    return str(uuid4())


class AutonomyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_steps: int = 0
    autonomous_steps_count: int = 0
    approvals_requested: int = 0
    retries_used: int = 0
    recovery_count_after_failure: int = 0
    human_intervention_events: int = 0
    human_intervention_ratio: float = 0.0
    irreversible_actions_attempted: int = 0


class StepTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    agent_name: str = "general"
    architecture_mode: str = ""
    phase: str = ""
    status: str = "in_progress"
    summary: str = ""
    selected_tools: list[str] = Field(default_factory=list)
    tool_statuses: dict[str, str] = Field(default_factory=dict)
    autonomous: bool = True
    approvals_requested: int = 0
    retry_triggered: bool = False
    recovered_after_failure: bool = False
    human_intervention_required: bool = False
    irreversible_actions_attempted: int = 0
    verification_status: str = "pending"
    reflection_status: str = "pending"
    termination_signal: str = "continue"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)


class RuntimeTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=_new_trace_id)
    request_id: str = ""
    state_id: str = ""
    user_id: str = ""
    session_id: str = ""
    objective: str = ""
    assigned_agent: str = "general"
    architecture_mode: str = ""
    termination_reason: str = ""
    summary: str = ""
    autonomy_metrics: AutonomyMetrics = Field(default_factory=AutonomyMetrics)
    steps: list[StepTrace] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=_utc_now)
