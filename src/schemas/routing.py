from __future__ import annotations

from datetime import datetime, timezone
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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RoutingRequest(BaseModel):
    task_type: ModelTaskType
    task_family: str | None = None
    selected_model: str | None = None
    selected_provider: ModelProvider | None = None
    require_structured_output: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingPolicyEntry(BaseModel):
    task_type: ModelTaskType
    provider: ModelProvider
    default_model: str
    rationale: str


class HistoricalRoutingStats(BaseModel):
    task_type: ModelTaskType
    task_family: str
    provider: ModelProvider
    sample_count: int = 0
    structured_output_validity_rate: float = 0.0
    average_latency_ms: int | None = None
    task_success_rate: float = 0.0
    average_completeness: float = 0.0
    retry_frequency: float = 0.0
    last_evaluated_at: datetime | None = None


class RoutingScore(BaseModel):
    task_type: ModelTaskType
    task_family: str
    provider: ModelProvider
    sample_count: int = 0
    structured_output_validity_component: float = 0.0
    latency_component: float = 0.0
    task_success_component: float = 0.0
    completeness_component: float = 0.0
    retry_penalty: float = 0.0
    total_score: float = 0.0
    eligible: bool = False
    rationale: str = ""


class AdaptiveRoutingDecision(BaseModel):
    task_type: ModelTaskType
    task_family: str
    selected_provider: ModelProvider
    fallback_provider: ModelProvider
    used_history: bool = False
    minimum_samples_required: int = 0
    scores: list[RoutingScore] = Field(default_factory=list)
    historical_stats: list[HistoricalRoutingStats] = Field(default_factory=list)
    reason: str = ""
    recorded_at: datetime = Field(default_factory=_utc_now)


class TaskFamilyRoutingWinner(BaseModel):
    task_type: ModelTaskType
    task_family: str
    selected_provider: ModelProvider
    fallback_provider: ModelProvider
    winning_score: float = 0.0
    sample_count: int = 0
    structured_output_validity_rate: float = 0.0
    task_success_rate: float = 0.0
    average_completeness: float = 0.0
    average_latency_ms: int | None = None
    retry_frequency: float = 0.0
    used_history: bool = False
    reason: str = ""
    last_evaluated_at: datetime | None = None


class RoutingDecision(BaseModel):
    task_type: ModelTaskType
    task_family: str = ""
    provider: ModelProvider
    model: str
    reason: str
    fallback_provider: ModelProvider | None = None
    adaptive_decision: AdaptiveRoutingDecision | None = None
    candidate_models: list[str] = Field(default_factory=list)
