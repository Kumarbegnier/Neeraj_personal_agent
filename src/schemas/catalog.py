from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_event_id() -> str:
    return str(uuid4())


class AgentDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    display_name: str
    role: str
    description: str
    responsibilities: list[str] = Field(default_factory=list)
    default_tools: list[str] = Field(default_factory=list)


class AgentCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: list[AgentDescriptor] = Field(default_factory=list)


class ToolDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    description: str
    risk_level: str = "low"
    side_effect: str = "none"
    structured_output: bool = True
    retryable: bool = True
    supports_dry_run: bool = True
    mcp_ready: bool = True
    contract_version: str = "tool-quality-v1"


class ToolCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: list[ToolDescriptor] = Field(default_factory=list)


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=_new_event_id)
    event: str
    payload: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=_utc_now)


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[AuditEvent] = Field(default_factory=list)
