from __future__ import annotations

from src.memory.episodic_store import EpisodicStore
from src.memory.manager import MemoryManager
from src.memory.semantic_store import SemanticStore
from src.runtime.models import MemoryRecord, MemorySnapshot

from agent_runtime.memory import MemorySystem


class MemoryRetrievalService:
    def __init__(
        self,
        memory_system: MemorySystem,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
    ) -> None:
        self._manager = MemoryManager(memory_system, episodic_store, semantic_store)

    def snapshot(self, user_id: str, session_id: str) -> MemorySnapshot:
        return self._manager.snapshot(user_id, session_id)

    def recent_context(self, user_id: str, session_id: str, limit: int = 6) -> MemorySnapshot:
        self._manager.hydrate_session(user_id, session_id, limit=limit)
        return self._manager.snapshot(user_id, session_id)

    def semantic_lookup(
        self,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 6,
    ) -> list[MemoryRecord]:
        return self._manager.semantic_lookup(user_id, session_id, query, limit)
