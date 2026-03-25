from __future__ import annotations

import unittest

from agent_runtime.context_engine import ContextEngine
from agent_runtime.handoff_engine import HandoffEngine
from agent_runtime.models import (
    AgentRoute,
    AgentState,
    AuthContext,
    Channel,
    ContextSignal,
    ContextSnapshot,
    ExecutionPlan,
    ExecutionResult,
    GatewayResult,
    MemoryRecord,
    MemorySnapshot,
    ModelEvaluationRecord,
    ModelExecutionRecord,
    ObservationRecord,
    ReActStepTrace,
    ToolResult,
    RateLimitStatus,
    ReflectionReport,
    StateTransition,
    TraceEvent,
    UserRequest,
    VerificationReport,
    WorkingMemory,
)
from agent_runtime.orchestrator import build_default_orchestrator
from src.schemas.context import ApprovalState
from src.schemas.catalog import ToolDescriptor


def build_gateway(normalized_message: str) -> GatewayResult:
    return GatewayResult(
        channel=Channel.text,
        normalized_message=normalized_message,
        auth=AuthContext(),
        rate_limit=RateLimitStatus(),
    )


def build_handoff_state() -> AgentState:
    request = UserRequest(
        message="Continue a long-running implementation task with compacted state.",
        goals=["ship_code", "verify_outputs"],
        metadata={"max_steps": 4},
    )
    gateway = build_gateway("continue a long-running implementation task with compacted state")
    memory = MemorySnapshot(
        summary="user: continue a long-running implementation task",
        retrieved=[
            MemoryRecord(
                memory_type="semantic",
                content="Previous attempt established the modular execution loop.",
                source="semantic_store",
                salience=0.9,
                tags=["architecture"],
            )
        ],
        goal_stack=["ship_code", "verify_outputs"],
        open_loops=["Resolve verification gaps before the next retry."],
        working_memory=WorkingMemory(
            objective="continue a long-running implementation task with compacted state",
            distilled_context="The task is mid-flight and needs a compact reusable handoff.",
            open_questions=["Which module should own the retry repair?"],
            retrieved_facts=["The modular execution loop already exists."],
            plan_checkpoint="react_iteration_2",
        ),
    )
    packet = ContextEngine(
        tool_descriptor_loader=lambda: [
            ToolDescriptor(name="working_memory", category="memory", description="Inspect working memory."),
            ToolDescriptor(name="generate_code", category="coding", description="Generate code."),
        ]
    ).build_context(
        request=request,
        gateway=gateway,
        memory_snapshot=memory,
        active_goals=["ship_code", "verify_outputs"],
        requested_capabilities=["coding", "verification"],
        constraints=["Keep the state compact and reusable."],
        approval_state=ApprovalState(
            permission_mode="confirm_required",
            requires_confirmation=True,
            risk_level="medium",
            approval_granted=False,
            blocked_tools=["deploy_code"],
            gated_actions=["confirmation_required"],
            rationale="Deployment-like actions still need approval.",
        ),
        current_execution_mode="react",
    )

    state = AgentState(request=request, gateway=gateway, step_index=2, max_steps=4)
    state.memory = memory
    state.context_packet = packet
    state.context = ContextSnapshot(
        user_id=request.user_id,
        session_id=request.session_id,
        channel=request.channel,
        latest_message=gateway.normalized_message,
        gateway=gateway,
        memory=memory,
        active_goals=["ship_code", "verify_outputs"],
        system_goals=["stay_safe"],
        constraints=["Keep the state compact and reusable."],
        requested_capabilities=["coding", "verification"],
        signals=ContextSignal(complexity="high", risk_level="medium"),
        context_packet=packet,
        execution_mode="react",
    )
    state.route = AgentRoute(agent_name="coding", rationale="Coding still owns this task.")
    state.plan = ExecutionPlan(
        objective=request.message,
        task_summary="Continue a long-running implementation safely.",
        subtasks=[],
        required_tools=["working_memory", "generate_code"],
        risk_level="medium",
        approval_needed=True,
        reasoning="The next iteration should use compacted state.",
        success_criteria=["Preserve continuity without replaying the entire verbose history."],
        failure_modes=["The next iteration loses the current blockers and next actions."],
        verification_focus=["State continuity remains intact."],
        decomposition_strategy="react",
    )
    state.execution = ExecutionResult(
        agent_name="coding",
        summary="Refined the runtime seam and identified the remaining repair path.",
        actions=["Updated the runtime seam", "Prepared the next repair path"],
        observations=["working_memory:objective aligned"],
        unresolved=["deploy_code:gated"],
        ready_for_response=False,
        requires_replan=True,
        next_focus="Resolve verification gaps before attempting a deploy-adjacent step.",
    )
    state.verification = VerificationReport(
        status="needs_attention",
        summary="Verification found one remaining continuity gap.",
        gaps=["The retry repair has not been implemented yet."],
        confidence=0.6,
        retry_recommended=True,
    )
    state.reflection = ReflectionReport(
        status="needs_attention",
        summary="Reflection recommended a compact handoff before the next retry.",
        issues=["The loop is carrying too much verbose state."],
        repairs=["Condense progress into a reusable handoff packet."],
        lessons=["Compact state should preserve blockers and next actions."],
        next_actions=["Resume from the compact handoff packet on the next iteration."],
        retry_recommended=True,
    )
    state.blocked_tools = ["deploy_code"]
    return state


class HandoffEngineTests(unittest.TestCase):
    def test_build_handoff_packet_captures_required_state(self) -> None:
        state = build_handoff_state()

        packet = HandoffEngine().build_handoff(state)

        self.assertIn("Completed 2 iteration(s)", packet.summary.completed_work_summary)
        self.assertTrue(packet.open_questions)
        self.assertIn("deploy_code", packet.blocked_tools)
        self.assertTrue(packet.approval_state.requires_confirmation)
        self.assertEqual(packet.memory_snapshot.goal_stack, ["ship_code", "verify_outputs"])
        self.assertTrue(packet.next_actions)

    def test_compact_state_trims_verbose_history_and_keeps_reusable_context(self) -> None:
        state = build_handoff_state()
        state.reasoning_notes = [f"note-{index}" for index in range(10)]
        state.trace = [TraceEvent(stage=f"stage-{index}", detail="detail") for index in range(30)]
        state.state_transitions = [
            StateTransition(step_index=index + 1, prior_status="x", next_status="y")
            for index in range(8)
        ]
        state.react_trace = [ReActStepTrace(step_index=index + 1) for index in range(5)]
        state.model_runs = [
            ModelExecutionRecord(
                task_type="reasoning",
                stage=f"stage-{index}",
                provider="stub",
                model="stub-model",
                status="completed",
            )
            for index in range(12)
        ]
        state.model_evaluations = [
            ModelEvaluationRecord(task_type="reasoning", provider="stub", model="stub-model")
            for _ in range(12)
        ]
        state.observations = [
            ObservationRecord(
                step_index=2,
                source=f"tool-{index}",
                summary=f"summary-{index}",
            )
            for index in range(12)
        ]
        state.tool_history = [
            ToolResult(
                tool_name=f"tool-{index}",
                status="success",
            )
            for index in range(14)
        ]
        state.handoff_packet = HandoffEngine().build_handoff(state)

        trimmed = HandoffEngine().compact_state(state)

        self.assertTrue(trimmed)
        self.assertLessEqual(len(state.observations), 6)
        self.assertLessEqual(len(state.tool_history), 12)
        self.assertLessEqual(len(state.reasoning_notes), 6)
        self.assertLessEqual(len(state.trace), 24)
        self.assertEqual(
            state.memory.working_memory.distilled_context,
            state.handoff_packet.reusable_context[:500],
        )
        self.assertEqual(
            state.context.metadata["handoff_packet"]["handoff_id"],
            state.handoff_packet.handoff_id,
        )

    def test_orchestrator_returns_handoff_packet_for_long_running_task(self) -> None:
        orchestrator = build_default_orchestrator()

        response = orchestrator.handle(
            UserRequest(
                message="Draft and send an email now, then keep iterating until the approval issue is resolved.",
                metadata={"max_steps": 4},
            )
        )

        self.assertIsNotNone(response.handoff_packet)
        self.assertTrue(response.loop_state.handoff_available)
        self.assertTrue(response.handoff_packet.summary.completed_work_summary)
        self.assertTrue(response.handoff_packet.next_actions or response.handoff_packet.open_questions)


if __name__ == "__main__":
    unittest.main()
