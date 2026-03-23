from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Protocol

from src.core.config import get_settings
from src.runtime.models import MemoryRecord


def _tokenize(text: str) -> dict[str, float]:
    frequencies: dict[str, float] = {}
    for token in re.findall(r"[a-z0-9_]+", text.lower()):
        if len(token) <= 2:
            continue
        frequencies[token] = frequencies.get(token, 0.0) + 1.0
    return frequencies


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in set(left) | set(right))
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class SemanticMemoryBackend(Protocol):
    def upsert(self, user_id: str, session_id: str, record: MemoryRecord) -> None:
        ...

    def search(
        self,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 6,
    ) -> list[MemoryRecord]:
        ...

    def health(self) -> dict[str, str | bool]:
        ...


class InMemoryFaissCompatibleStore:
    """A FAISS-compatible abstraction with a lightweight in-memory implementation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._records: dict[str, list[MemoryRecord]] = defaultdict(list)

    def _key(self, user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    def upsert(self, user_id: str, session_id: str, record: MemoryRecord) -> None:
        key = self._key(user_id, session_id)
        existing = self._records[key]
        for index, candidate in enumerate(existing):
            if candidate.memory_id == record.memory_id or candidate.content == record.content:
                existing[index] = record
                return
        existing.append(record)

    def search(
        self,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 6,
    ) -> list[MemoryRecord]:
        key = self._key(user_id, session_id)
        query_vector = _tokenize(query)
        ranked = sorted(
            (
                (
                    _cosine_similarity(query_vector, _tokenize(record.content)) + record.salience,
                    record,
                )
                for record in self._records.get(key, [])
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        results: list[MemoryRecord] = []
        for score, record in ranked[:limit]:
            results.append(record.model_copy(update={"score": round(score, 4)}))
        return results

    def health(self) -> dict[str, str | bool]:
        return {
            "backend": self.settings.semantic_backend,
            "connected": True,
        }


class SemanticStore:
    def __init__(self, backend: SemanticMemoryBackend | None = None) -> None:
        self._backend = backend or InMemoryFaissCompatibleStore()

    def save(self, user_id: str, session_id: str, record: MemoryRecord) -> None:
        self._backend.upsert(user_id, session_id, record)

    def search(
        self,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 6,
    ) -> list[MemoryRecord]:
        return self._backend.search(user_id, session_id, query, limit)

    def recent(
        self,
        user_id: str,
        session_id: str,
        limit: int = 6,
    ) -> list[MemoryRecord]:
        return self.search(user_id, session_id, query="recent context", limit=limit)

    def health(self) -> dict[str, str | bool]:
        return self._backend.health()
