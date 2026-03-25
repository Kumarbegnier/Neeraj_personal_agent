from __future__ import annotations

from typing import Any

from src.runtime.models import RuntimeTrace, UserRequest
from src.safety.audit import AuditService


class ObservabilityService:
    """Persists typed runtime traces and exposes recent observability snapshots."""

    def __init__(
        self,
        audit_service: AuditService,
        *,
        max_traces: int = 100,
    ) -> None:
        self._audit_service = audit_service
        self._max_traces = max_traces
        self._runtime_traces: list[RuntimeTrace] = []

    def persist_runtime_trace(
        self,
        request: UserRequest,
        runtime_trace: RuntimeTrace | None,
    ) -> None:
        if runtime_trace is None:
            return

        stored_trace = runtime_trace.model_copy(deep=True)
        self._runtime_traces.append(stored_trace)
        self._runtime_traces = self._runtime_traces[-self._max_traces :]
        self._audit_service.record(
            "runtime_trace",
            {
                "request_id": stored_trace.request_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "assigned_agent": stored_trace.assigned_agent,
                "termination_reason": stored_trace.termination_reason,
                "autonomy_metrics": stored_trace.autonomy_metrics.model_dump(mode="json"),
                "step_count": len(stored_trace.steps),
                "trace": stored_trace.model_dump(mode="json"),
            },
        )

    def recent_runtime_traces(self, limit: int = 25) -> list[RuntimeTrace]:
        bounded_limit = max(1, limit)
        if self._runtime_traces:
            return [trace.model_copy(deep=True) for trace in self._runtime_traces[-bounded_limit:]]
        return self._load_from_audit(limit=bounded_limit)

    def health(self) -> dict[str, Any]:
        latest = self.recent_runtime_traces(limit=1)
        latest_trace = latest[-1] if latest else None
        return {
            "buffered_trace_count": len(self._runtime_traces),
            "available": latest_trace is not None,
            "latest_request_id": latest_trace.request_id if latest_trace is not None else None,
            "latest_recorded_at": (
                latest_trace.recorded_at.isoformat() if latest_trace is not None else None
            ),
            "latest_autonomous_steps": (
                latest_trace.autonomy_metrics.autonomous_steps_count if latest_trace is not None else 0
            ),
        }

    def _load_from_audit(self, *, limit: int) -> list[RuntimeTrace]:
        traces: list[RuntimeTrace] = []
        for event in self._audit_service.recent(limit=max(limit * 6, limit)):
            if event.event != "runtime_trace":
                continue
            payload = event.payload.get("trace")
            if not isinstance(payload, dict):
                continue
            try:
                traces.append(RuntimeTrace.model_validate(payload))
            except Exception:
                continue
        return [trace.model_copy(deep=True) for trace in traces[-limit:]]
