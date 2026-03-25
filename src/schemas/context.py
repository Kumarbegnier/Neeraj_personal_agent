from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .catalog import ToolDescriptor


class NormalizedUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_message: str
    normalized_message: str
    channel: str
    goals: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextEpisodicMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    timestamp: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextSemanticMemoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str | None = None
    memory_type: str = "semantic"
    content: str
    source: str = ""
    score: float = 0.0
    salience: float = 0.0
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ContextMemorySlice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = "No prior conversation history."
    retrieval_query: str = ""
    recent_episodic_memory: list[ContextEpisodicMemory] = Field(default_factory=list)
    semantic_memory_results: list[ContextSemanticMemoryResult] = Field(default_factory=list)
    working_memory_summary: str = ""
    open_loops: list[str] = Field(default_factory=list)
    goal_stack: list[str] = Field(default_factory=list)
    retrieved_facts: list[str] = Field(default_factory=list)


class ToolCapability(ToolDescriptor):
    model_config = ConfigDict(extra="forbid")


class ToolAvailabilitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available_tools: list[ToolCapability] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    risky_tools: list[str] = Field(default_factory=list)
    approval_gated_tools: list[str] = Field(default_factory=list)
    total_tools: int = 0


class ApprovalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    permission_mode: str = "auto_approved"
    requires_confirmation: bool = False
    risk_level: str = "low"
    approval_granted: bool = False
    blocked_tools: list[str] = Field(default_factory=list)
    gated_actions: list[str] = Field(default_factory=list)
    rationale: str = ""


class ProviderRouteHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: str
    provider: str | None = None
    model: str | None = None
    reason: str = ""
    candidate_models: list[str] = Field(default_factory=list)


class ProviderRoutingHints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hints: list[ProviderRouteHint] = Field(default_factory=list)


class ContextPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_user_request: NormalizedUserRequest
    memory: ContextMemorySlice
    active_goals: list[str] = Field(default_factory=list)
    requested_capabilities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    tool_availability: ToolAvailabilitySnapshot = Field(default_factory=ToolAvailabilitySnapshot)
    approval_state: ApprovalState = Field(default_factory=ApprovalState)
    provider_routing_hints: ProviderRoutingHints = Field(default_factory=ProviderRoutingHints)
    current_execution_mode: str = "react"
    context_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

