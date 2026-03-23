from __future__ import annotations

from src.core.config import Settings, get_settings
from src.memory.episodic_store import EpisodicStore
from src.memory.semantic_store import SemanticStore
from src.runtime.models import InteractionResponse, MemoryRecord, MemorySnapshot, SessionState, UserRequest

from agent_runtime.memory import MemorySystem


class MemoryManager:
    """Coordinates runtime memory hydration with durable episodic and semantic stores."""

    def __init__(
        self,
        memory_system: MemorySystem,
        episodic_store: EpisodicStore,
        semantic_store: SemanticStore,
        settings: Settings | None = None,
    ) -> None:
        self._memory_system = memory_system
        self._episodic_store = episodic_store
        self._semantic_store = semantic_store
        self._settings = settings or get_settings()

    def hydrate_request(self, request: UserRequest) -> None:
        self.hydrate_session(request.user_id, request.session_id)
        semantic_hits = self._semantic_store.search(
            request.user_id,
            request.session_id,
            query=request.message,
            limit=self._settings.semantic_top_k,
        )
        if semantic_hits:
            self._memory_system.ingest_semantic_records(
                request.user_id,
                request.session_id,
                semantic_hits,
            )

    def hydrate_session(self, user_id: str, session_id: str, limit: int = 6) -> None:
        durable_turns = self._episodic_store.load_recent_context(user_id, session_id, limit=limit)
        if durable_turns:
            self._memory_system.ingest_history(user_id, session_id, durable_turns)

    def session_state(self, user_id: str, session_id: str) -> SessionState:
        self.hydrate_session(user_id, session_id)
        return self._memory_system.get_session_state(user_id, session_id)

    def snapshot(self, user_id: str, session_id: str) -> MemorySnapshot:
        return self._memory_system.build_snapshot(user_id, session_id)

    def semantic_lookup(
        self,
        user_id: str,
        session_id: str,
        query: str,
        limit: int = 6,
    ) -> list[MemoryRecord]:
        records = self._semantic_store.search(user_id, session_id, query, limit)
        if records:
            self._memory_system.ingest_semantic_records(user_id, session_id, records)
        return records

    def persist_interaction(
        self,
        request: UserRequest,
        response: InteractionResponse,
    ) -> None:
        self._episodic_store.save_interaction(
            request_id=response.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            request_text=request.message,
            response_text=response.response,
            metadata={
                "assigned_agent": response.assigned_agent,
                "termination_reason": response.termination_reason,
                "approval_mode": response.safety.permission.mode.value,
            },
            trace=[event.model_dump() for event in response.trace],
        )
        self._episodic_store.store_task_outcome(
            user_id=request.user_id,
            session_id=request.session_id,
            task_name=request.message[:80],
            outcome=response.termination_reason,
            metadata={
                "assigned_agent": response.assigned_agent,
                "verification_status": response.verification.status,
                "loop_count": response.loop_count,
            },
        )
        for record in self._semantic_records_for(response):
            self._semantic_store.save(request.user_id, request.session_id, record)

    def _semantic_records_for(self, response: InteractionResponse) -> list[MemoryRecord]:
        return [
            MemoryRecord(
                memory_type="semantic",
                content=f"Interaction summary: {response.response[:240]}",
                source="memory_manager",
                salience=0.75,
                tags=[response.assigned_agent, response.termination_reason],
            ),
            MemoryRecord(
                memory_type="semantic",
                content=f"Verification summary: {response.verification.summary}",
                source="memory_manager",
                salience=0.7,
                tags=["verification"],
            ),
        ]
