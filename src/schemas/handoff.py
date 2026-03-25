from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .context import ApprovalState, ContextMemorySlice


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_handoff_id() -> str:
    return str(uuid4())


class OpenQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    source: str = "working_memory"
    blocking: bool = False
    suggested_owner: str = "general"
    related_tools: list[str] = Field(default_factory=list)


class HandoffSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str
    completed_work_summary: str
    completed_steps: list[str] = Field(default_factory=list)
    current_status: str = "in_progress"
    verification_summary: str = ""
    reflection_summary: str = ""


class HandoffPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    handoff_id: str = Field(default_factory=_new_handoff_id)
    created_at: datetime = Field(default_factory=_utc_now)
    summary: HandoffSummary
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    approval_state: ApprovalState = Field(default_factory=ApprovalState)
    memory_snapshot: ContextMemorySlice = Field(default_factory=ContextMemorySlice)
    source_agent: str = "general"
    target_agent: str | None = None
    architecture_mode: str = ""
    loop_iteration: int = 0
    reusable_context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
