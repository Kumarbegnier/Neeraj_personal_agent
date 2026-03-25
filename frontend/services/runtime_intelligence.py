from __future__ import annotations

from dataclasses import dataclass

from src.runtime.models import RuntimeTrace, SessionState
from src.schemas.platform import HealthResponse
from src.schemas.routing import TaskFamilyRoutingWinner

from .api_client import ApiClient


@dataclass(frozen=True)
class RuntimeIntelligenceSnapshot:
    health: HealthResponse
    session_snapshot: SessionState
    runtime_traces: list[RuntimeTrace]
    evaluation_winners: list[TaskFamilyRoutingWinner]


class RuntimeIntelligenceService:
    def __init__(self, client: ApiClient) -> None:
        self._client = client

    def load(
        self,
        *,
        user_id: str,
        session_id: str,
        trace_limit: int = 25,
        winners_limit: int = 12,
    ) -> RuntimeIntelligenceSnapshot:
        return RuntimeIntelligenceSnapshot(
            health=self._client.health(),
            session_snapshot=self._client.session_state(user_id=user_id, session_id=session_id),
            runtime_traces=self._client.runtime_traces(limit=trace_limit),
            evaluation_winners=self._client.evaluation_winners(limit=winners_limit),
        )
