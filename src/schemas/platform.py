from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.runtime.models import (
    GatewayHeaders,
    InteractionResponse,
    MemorySnapshot,
    PermissionDecision,
    SkillDescriptor,
    TaskGraph,
    TraceEvent,
    UserRequest,
)
from src.runtime.workflow import StageDescriptor
from src.schemas.plan import ControlDecision, ExecutionPlan


class HealthResponse(BaseModel):
    name: str
    status: str
    environment: str
    llm: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    semantic_memory: dict[str, Any] = Field(default_factory=dict)
    audit: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(UserRequest):
    pass


class ChatResponse(BaseModel):
    interaction: InteractionResponse


class PlanRequest(UserRequest):
    pass


class PlanResponse(BaseModel):
    request_id: str
    state_id: str
    assigned_agent: str
    control: ControlDecision
    plan: ExecutionPlan
    task_graph: TaskGraph
    skills: list[SkillDescriptor] = Field(default_factory=list)
    memory: MemorySnapshot
    permission: PermissionDecision
    trace: list[TraceEvent] = Field(default_factory=list)


class ArchitectureResponse(BaseModel):
    stages: list[StageDescriptor] = Field(default_factory=list)


class ExecuteRequest(UserRequest):
    headers: GatewayHeaders | None = None


class ExecuteResponse(BaseModel):
    interaction: InteractionResponse
