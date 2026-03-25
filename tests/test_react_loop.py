from __future__ import annotations

import unittest

from agent_runtime.orchestrator import build_default_orchestrator
from agent_runtime.models import (
    ActionDecision,
    AgentDecision,
    AgentRoute,
    AgentState,
    ExecutionPlan,
    ExecutionResult,
    VerificationResult,
    ReasoningStep,
    ToolRequest,
    VerificationReport,
    UserRequest,
)
from src.schemas.adaptive import LoopPhase
from agent_runtime.reasoning import ReasoningEngine
from agent_runtime.stopping import StoppingEngine
from agent_runtime.tool_selection import ToolSelectionEngine


def build_state() -> AgentState:
    state = AgentState(
        request=UserRequest(message="Build a multi-step agent loop."),
    )
    state.plan = ExecutionPlan(
        objective=state.request.message,
        task_summary="Deliver a multi-step agent loop.",
        subtasks=[],
        required_tools=["working_memory", "plan_analyzer", "generate_code"],
        risk_level="low",
        approval_needed=False,
        success_criteria=["The loop can reason, act, reflect, and stop cleanly."],
        reasoning="Use a ReAct-style execution flow.",
        react_cycles=[],
        steps=[],
        assumptions=[],
        constraints=[],
        failure_modes=[],
        verification_focus=[],
        decomposition_strategy="react",
    )
    state.route = AgentRoute(agent_name="coding", rationale="Coding best fits the request.")
    state.decision = AgentDecision(
        agent_name="coding",
        summary="Prepare candidate tool calls for the next step.",
        skill_names=["architecture"],
        tool_requests=[
            ToolRequest(tool_name="working_memory", purpose="Load current context.", priority=1),
            ToolRequest(tool_name="plan_analyzer", purpose="Inspect plan state.", priority=2),
            ToolRequest(tool_name="generate_code", purpose="Draft implementation.", priority=3),
        ],
        expected_deliverables=["Fresh evidence for the current coding objective."],
    )
    return state


class ReActLoopTests(unittest.TestCase):
    def test_action_and_verification_schema_names_are_available(self) -> None:
        action = ActionDecision(
            selected_tool_names=["working_memory"],
            deferred_tool_names=["generate_code"],
            rationale="Ground context before drafting code.",
            expected_outcome="Fresh memory evidence.",
        )
        verification = VerificationResult(
            status="passed",
            summary="Evidence was grounded.",
            verified_claims=["The loop produced grounded evidence."],
        )

        self.assertEqual(action.selected_tool_names, ["working_memory"])
        self.assertEqual(verification.status, "passed")

    def test_reasoning_fallback_uses_candidate_tools(self) -> None:
        state = build_state()

        reasoning = ReasoningEngine().reason(state)

        self.assertEqual(reasoning.candidate_tools, ["working_memory", "plan_analyzer", "generate_code"])
        self.assertFalse(reasoning.should_replan)
        self.assertIn("reason over", reasoning.reasoning_summary.lower())

    def test_tool_selection_respects_reasoning_and_blocked_tools(self) -> None:
        state = build_state()
        state.blocked_tools = ["generate_code"]
        state.reasoning = ReasoningStep(
            objective=state.request.message,
            thought="Prefer grounding tools before code generation.",
            reasoning_summary="Focus on memory and plan evidence first.",
            action_strategy="Start with the highest-signal low-risk tools.",
            candidate_tools=["plan_analyzer", "working_memory", "generate_code"],
            selected_skills=["architecture"],
            expected_observation="Grounded plan evidence.",
        )

        selection, selected_requests = ToolSelectionEngine().select(state)

        self.assertEqual(selection.selected_tool_names[:2], ["plan_analyzer", "working_memory"])
        self.assertNotIn("generate_code", selection.selected_tool_names)
        self.assertEqual([request.tool_name for request in selected_requests], ["plan_analyzer", "working_memory"])

    def test_stopping_engine_stops_on_verified_ready_state(self) -> None:
        state = build_state()
        state.execution = ExecutionResult(
            agent_name="coding",
            summary="Produced enough evidence to finalize the answer.",
            ready_for_response=True,
            goal_status="ready",
        )
        state.verification = VerificationReport(
            status="passed",
            summary="Verification passed.",
            confidence=0.9,
        )

        decision = StoppingEngine().decide(state)

        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.trigger, "goal_achieved")

    def test_stopping_engine_requests_replan_when_state_demands_it(self) -> None:
        state = build_state()
        state.needs_replan = True
        state.reasoning = ReasoningStep(
            objective=state.request.message,
            thought="The current candidate set cannot progress.",
            reasoning_summary="Reasoning found a planning gap.",
            action_strategy="Replan before acting.",
            should_replan=True,
            replan_reason="No viable next action was identified.",
        )

        decision = StoppingEngine().decide(state)

        self.assertFalse(decision.should_stop)
        self.assertTrue(decision.requires_replan)
        self.assertEqual(decision.trigger, "replan")

    def test_stopping_engine_halts_when_retry_budget_is_exhausted(self) -> None:
        state = build_state()
        state.needs_retry = True
        state.retry_count = 1
        state.loop_state.retry_budget = 1
        state.reasoning = ReasoningStep(
            objective=state.request.message,
            thought="The loop needs one more try but has no retry budget left.",
            reasoning_summary="Retry budget should stop the loop.",
            action_strategy="Stop instead of replanning.",
            should_replan=True,
            replan_reason="Retry budget exceeded.",
        )

        decision = StoppingEngine().decide(state)

        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.trigger, "retry_budget_exhausted")

    def test_orchestrator_records_loop_state_and_step_trace(self) -> None:
        orchestrator = build_default_orchestrator()

        response = orchestrator.handle(UserRequest(message="Build a small agent loop test."))

        self.assertEqual(response.loop_state.phase, LoopPhase.COMPLETE)
        self.assertGreaterEqual(response.loop_state.retry_budget, 0)
        self.assertIsNotNone(response.architecture)
        self.assertEqual(response.control.architecture_mode, response.architecture.mode.value)
        self.assertEqual(
            response.plan.decomposition_strategy,
            response.architecture.loop_strategy.replace("_", " "),
        )
        self.assertGreaterEqual(len(response.react_trace), 1)
        self.assertTrue(response.loop_state.memory_checkpoints)
        self.assertGreaterEqual(len(response.step_traces), 1)
        self.assertIsNotNone(response.runtime_trace)
        self.assertEqual(response.autonomy_metrics.total_steps, len(response.step_traces))
        self.assertEqual(response.autonomy_metrics.retries_used, response.loop_state.retry_count)
        self.assertEqual(response.runtime_trace.autonomy_metrics.total_steps, response.autonomy_metrics.total_steps)
        self.assertEqual(len(response.runtime_trace.steps), len(response.step_traces))
        self.assertEqual(
            response.react_trace[0].architecture_mode,
            response.architecture.mode.value,
        )
        self.assertTrue(response.react_trace[0].architecture_summary)
        self.assertTrue(response.react_trace[0].verification_summary)
        self.assertTrue(
            {
                LoopPhase.OBSERVE,
                LoopPhase.SELECT_ARCHITECTURE,
                LoopPhase.REASON,
                LoopPhase.ACT,
                LoopPhase.VERIFY,
                LoopPhase.REFLECT,
                LoopPhase.LOOP_CONTROL,
            }.issubset(set(response.react_trace[0].loop_phases))
        )


if __name__ == "__main__":
    unittest.main()
