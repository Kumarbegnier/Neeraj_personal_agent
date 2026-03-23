from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_event_id() -> str:
    return str(uuid4())


class AgentDescriptor(BaseModel):
    key: str
    display_name: str
    role: str
    description: str
    responsibilities: list[str] = Field(default_factory=list)
    default_tools: list[str] = Field(default_factory=list)


class AgentCatalog(BaseModel):
    agents: list[AgentDescriptor] = Field(default_factory=list)


class ToolDescriptor(BaseModel):
    name: str
    category: str
    description: str
    risk_level: str = "low"
    side_effect: str = "none"
    structured_output: bool = True


class ToolCatalog(BaseModel):
    tools: list[ToolDescriptor] = Field(default_factory=list)


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=_new_event_id)
    event: str
    payload: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=_utc_now)


class AuditLogResponse(BaseModel):
    events: list[AuditEvent] = Field(default_factory=list)

