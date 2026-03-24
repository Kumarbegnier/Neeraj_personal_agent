from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


class ModelTaskType(str, Enum):
    ORCHESTRATION = "orchestration"
    TOOL_EXECUTION = "tool_execution"
    COMMUNICATION = "communication"
    RESEARCH = "research"
    WEB_GROUNDING = "web_grounding"
    PLANNING = "planning"
    REASONING = "reasoning"
    REFLECTION = "reflection"


class RoutingRequest(BaseModel):
    task_type: ModelTaskType
    selected_model: str | None = None
    selected_provider: ModelProvider | None = None
    require_structured_output: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingPolicyEntry(BaseModel):
    task_type: ModelTaskType
    provider: ModelProvider
    default_model: str
    rationale: str


class RoutingDecision(BaseModel):
    task_type: ModelTaskType
    provider: ModelProvider
    model: str
    reason: str
    candidate_models: list[str] = Field(default_factory=list)
