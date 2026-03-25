from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .routing import ModelProvider, ModelTaskType


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProviderEvaluationResult(BaseModel):
    task_type: ModelTaskType
    task_family: str = ""
    provider: ModelProvider
    model: str
    structured_output_validity: bool = False
    latency_ms: int = 0
    task_success: bool = False
    response_completeness: float = 0.0
    retry_count: int = 0
    notes: list[str] = Field(default_factory=list)
    used_fallback: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=_utc_now)


class EvaluationSummary(BaseModel):
    task_type: ModelTaskType
    task_family: str = ""
    results: list[ProviderEvaluationResult] = Field(default_factory=list)
    winner: ModelProvider | None = None
    summary: str = ""
