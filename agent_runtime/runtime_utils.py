from __future__ import annotations

import re
from collections.abc import Iterable

from src.services.modeling.types import StructuredGenerationResult

from .models import AgentState, ModelEvaluationRecord, ModelExecutionRecord, TraceEvent


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def recent_step_indexes(step_index: int) -> set[int]:
    return {step_index, max(0, step_index - 1)}


def recent_observation_summaries(state: AgentState, limit: int | None = None) -> list[str]:
    summaries = [
        observation.summary
        for observation in state.observations
        if observation.step_index in recent_step_indexes(state.step_index)
    ]
    return summaries if limit is None else summaries[-limit:]


def join_non_empty(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part)


def lowercase_surface(parts: Iterable[str]) -> str:
    return " ".join(part.lower() for part in parts if part)


def tokenize_words(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 2}


def selected_model_override(state: AgentState) -> str | None:
    selected = state.request.preferences.get("selected_model")
    if not selected:
        return None
    return str(selected).strip() or None


def record_model_result(state: AgentState, result: StructuredGenerationResult) -> None:
    state.model_runs.append(
        ModelExecutionRecord(
            task_type=result.run.task_type,
            stage=result.run.stage,
            provider=result.run.provider,
            model=result.run.model,
            status=result.run.status,
            source=result.run.source,
            latency_ms=result.run.latency_ms,
            used_fallback=result.run.used_fallback,
            reason=result.run.reason,
            candidate_models=list(result.run.candidate_models),
            metadata=dict(result.run.metadata),
        )
    )
    state.model_evaluations.append(
        ModelEvaluationRecord(
            task_type=result.evaluation.task_type,
            provider=result.evaluation.provider,
            model=result.evaluation.model,
            score=result.evaluation.score,
            notes=list(result.evaluation.notes),
            compared_models=list(result.evaluation.compared_models),
            metadata=dict(result.evaluation.metadata),
        )
    )
    state.trace.append(
        TraceEvent(
            stage=f"Model Runtime / {result.run.stage}",
            detail=(
                f"Task '{result.run.task_type}' routed to provider '{result.run.provider}' "
                f"using model '{result.run.model}' with source '{result.run.source}'."
            ),
            payload={
                "status": result.run.status,
                "used_fallback": result.run.used_fallback,
                "latency_ms": result.run.latency_ms,
                "reason": result.run.reason,
                "candidate_models": list(result.run.candidate_models),
            },
        )
    )
