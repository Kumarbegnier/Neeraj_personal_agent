from __future__ import annotations

import unittest

from agent_runtime.models import (
    ActionDecision,
    AgentRoute,
    AgentState,
    PermissionDecision,
    PermissionMode,
    SessionPermissionState,
    StopDecision,
    ToolResult,
    UserRequest,
    VerificationResult,
)
from agent_runtime.observability import RuntimeObservabilityEngine
from src.schemas.adaptive import LoopPhase
from src.schemas.catalog import AuditEvent
from src.schemas.observability import AutonomyMetrics, RuntimeTrace, StepTrace
from src.services.observability_service import ObservabilityService


class FakeAuditService:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: str, payload: dict) -> None:
        self.events.append(AuditEvent(event=event, payload=payload))

    def recent(self, limit: int = 100) -> list[AuditEvent]:
        return list(self.events[-limit:])

    def health(self) -> dict[str, object]:
        return {"recent_event_count": len(self.events)}


class RuntimeObservabilityTests(unittest.TestCase):
    def test_runtime_observability_engine_tracks_autonomy_metrics(self) -> None:
        state = AgentState(request=UserRequest(message="Send the status update email."))
        state.step_index = 1
        state.loop_state.phase = LoopPhase.LOOP_CONTROL
        state.session = SessionPermissionState(
            user_id=state.request.user_id,
            session_id=state.request.session_id,
            permission=PermissionDecision(
                mode=PermissionMode.confirm_required,
                requires_confirmation=True,
                reason="Outbound actions require confirmation.",
            ),
        )
        state.route = AgentRoute(agent_name="communication", rationale="Communication is the right lane.")
        state.tool_selection = ActionDecision(
            selected_tool_names=["send_message"],
            rationale="Attempt the outbound send after drafting.",
        )
        state.last_tool_results = [
            ToolResult(
                tool_name="send_message",
                status="gated",
                risk_level="high",
                side_effect="send",
                blocked_reason="confirmation_required",
                output={"requires_confirmation": True},
            )
        ]
        state.verification = VerificationResult(
            status="needs_attention",
            summary="Execution is waiting on approval.",
            retry_recommended=True,
        )
        state.stop_decision = StopDecision(
            should_stop=False,
            trigger="replan",
            reason="Approval is required before the send can continue.",
            requires_replan=True,
        )
        state.needs_retry = True

        engine = RuntimeObservabilityEngine()
        step_trace = engine.record_iteration(state)

        self.assertEqual(step_trace.status, "needs_approval")
        self.assertEqual(step_trace.approvals_requested, 1)
        self.assertFalse(step_trace.autonomous)
        self.assertEqual(step_trace.irreversible_actions_attempted, 1)
        self.assertEqual(state.autonomy_metrics.total_steps, 1)
        self.assertEqual(state.autonomy_metrics.approvals_requested, 1)
        self.assertEqual(state.autonomy_metrics.irreversible_actions_attempted, 1)

    def test_observability_service_persists_and_rehydrates_runtime_traces(self) -> None:
        audit_service = FakeAuditService()
        service = ObservabilityService(audit_service)
        trace = RuntimeTrace(
            request_id="req-1",
            state_id="state-1",
            user_id="user-1",
            session_id="session-1",
            objective="Track observability",
            assigned_agent="general",
            autonomy_metrics=AutonomyMetrics(total_steps=1, autonomous_steps_count=1),
            steps=[StepTrace(step_index=1, phase="loop_control", status="completed", summary="Finished cleanly.")],
        )

        service.persist_runtime_trace(UserRequest(user_id="user-1", session_id="session-1", message="Observe"), trace)

        recent = service.recent_runtime_traces(limit=1)
        mirrored = ObservabilityService(audit_service).recent_runtime_traces(limit=1)

        self.assertEqual(recent[0].request_id, "req-1")
        self.assertEqual(mirrored[0].request_id, "req-1")
        self.assertTrue(service.health()["available"])


if __name__ == "__main__":
    unittest.main()
