from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel


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


@dataclass(frozen=True)
class ModelRoute:
    task_type: ModelTaskType
    provider: ModelProvider
    model: str
    reason: str
    candidate_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationTelemetry:
    task_type: str
    stage: str
    provider: str
    model: str
    status: str
    source: str
    latency_ms: int
    used_fallback: bool
    reason: str
    candidate_models: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationTelemetry:
    task_type: str
    provider: str
    model: str
    score: float
    notes: tuple[str, ...] = ()
    compared_models: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class StructuredGenerationResult(Generic[T]):
    output: T
    route: ModelRoute
    run: GenerationTelemetry
    evaluation: EvaluationTelemetry
