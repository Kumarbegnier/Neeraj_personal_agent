from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from src.schemas.routing import ModelProvider, ModelTaskType, RoutingDecision


ModelRoute = RoutingDecision


class GenerationTelemetry(BaseModel):
    task_type: str
    stage: str
    provider: str
    model: str
    status: str
    source: str
    latency_ms: int
    used_fallback: bool
    reason: str
    candidate_models: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationTelemetry(BaseModel):
    task_type: str
    task_family: str = ""
    provider: str
    model: str
    score: float
    notes: list[str] = Field(default_factory=list)
    compared_models: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured_output_validity: bool | None = None
    latency_ms: int | None = None
    task_success: bool | None = None
    response_completeness: float | None = None
    retry_count: int | None = None


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class StructuredGenerationResult(Generic[T]):
    output: T
    route: ModelRoute
    run: GenerationTelemetry
    evaluation: EvaluationTelemetry
