from __future__ import annotations

from src.memory.manager import MemoryManager
from src.runtime.models import AgentState, InteractionResponse, RuntimeTrace, SessionState, UserRequest
from src.safety.audit import AuditService
from src.schemas.catalog import AuditLogResponse
from src.services.observability_service import ObservabilityService


class RuntimeLifecycleService:
    """Handles request hydration, persistence, and audit capture around the runtime loop."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        audit_service: AuditService,
        observability_service: ObservabilityService,
    ) -> None:
        self._memory_manager = memory_manager
        self._audit_service = audit_service
        self._observability_service = observability_service

    def prepare_request(self, request: UserRequest) -> None:
        self._memory_manager.hydrate_request(request)

    def session_state(self, user_id: str, session_id: str) -> SessionState:
        return self._memory_manager.session_state(user_id, session_id)

    def record_plan_preview(self, request: UserRequest, state: AgentState) -> None:
        self._audit_service.record(
            "plan_preview",
            {
                "user_id": request.user_id,
                "session_id": request.session_id,
                "state_id": state.state_id,
                "assigned_agent": state.route.agent_name if state.route else "general",
                "model_run_count": len(state.model_runs),
            },
        )
        self._record_model_events(state.model_runs, state.model_evaluations)

    def finalize_interaction(
        self,
        request: UserRequest,
        response: InteractionResponse,
    ) -> None:
        self._memory_manager.persist_interaction(request, response)
        self._observability_service.persist_runtime_trace(request, response.runtime_trace)
        self._record_trace_events(request, response)
        self._record_tool_events(response)
        self._record_model_events(response.model_runs, response.model_evaluations)

    def recent_audit_events(self, limit: int = 100) -> AuditLogResponse:
        return AuditLogResponse(events=self._audit_service.recent(limit=limit))

    def recent_runtime_traces(self, limit: int = 25) -> list[RuntimeTrace]:
        return self._observability_service.recent_runtime_traces(limit=limit)

    def observability_health(self) -> dict[str, object]:
        return self._observability_service.health()

    def _record_trace_events(
        self,
        request: UserRequest,
        response: InteractionResponse,
    ) -> None:
        for event in response.trace:
            self._audit_service.record(
                "agent_step",
                {
                    "request_id": response.request_id,
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "stage": event.stage,
                    "detail": event.detail,
                    "payload": event.payload,
                },
            )

    def _record_tool_events(self, response: InteractionResponse) -> None:
        for tool_result in response.tool_results:
            self._audit_service.record(
                "tool_call",
                {
                    "request_id": response.request_id,
                    "tool_name": tool_result.tool_name,
                    "status": tool_result.status,
                    "risk_level": tool_result.risk_level,
                    "blocked_reason": tool_result.blocked_reason,
                },
            )

    def _record_model_events(self, model_runs, model_evaluations) -> None:
        for run in model_runs:
            self._audit_service.record(
                "model_invocation",
                {
                    "task_type": run.task_type,
                    "stage": run.stage,
                    "provider": run.provider,
                    "model": run.model,
                    "status": run.status,
                    "source": run.source,
                    "latency_ms": run.latency_ms,
                    "used_fallback": run.used_fallback,
                },
            )
        for evaluation in model_evaluations:
            self._audit_service.record(
                "model_evaluation",
                {
                    "task_type": evaluation.task_type,
                    "provider": evaluation.provider,
                    "model": evaluation.model,
                    "score": evaluation.score,
                    "notes": evaluation.notes,
                    "compared_models": evaluation.compared_models,
                    "structured_output_validity": evaluation.structured_output_validity,
                    "latency_ms": evaluation.latency_ms,
                    "task_success": evaluation.task_success,
                    "response_completeness": evaluation.response_completeness,
                },
            )
