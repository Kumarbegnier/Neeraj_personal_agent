from __future__ import annotations

import re
from collections.abc import Iterable

from .models import AgentState


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
