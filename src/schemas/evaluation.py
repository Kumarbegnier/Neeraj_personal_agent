from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .routing import ModelProvider, ModelTaskType


class ProviderEvaluationResult(BaseModel):
    provider: ModelProvider
    model: str
    structured_output_validity: bool = False
    latency_ms: int = 0
    task_success: bool = False
    response_completeness: float = 0.0
    notes: list[str] = Field(default_factory=list)
    used_fallback: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    task_type: ModelTaskType
    results: list[ProviderEvaluationResult] = Field(default_factory=list)
    winner: ModelProvider | None = None
    summary: str = ""
