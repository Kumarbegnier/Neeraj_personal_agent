from __future__ import annotations

from .models import AgentState, StopDecision


class StoppingEngine:
    """Deterministic stop policy for multi-step ReAct execution."""

    def decide(self, state: AgentState) -> StopDecision:
        if (
            self._criterion_enabled(state, "goal_achieved")
            and state.execution
            and state.execution.ready_for_response
            and not state.needs_retry
            and not state.needs_replan
        ):
            verification_passed = state.verification.status == "passed" if state.verification else True
            if verification_passed:
                return StopDecision(
                    should_stop=True,
                    trigger="goal_achieved",
                    reason="Execution produced enough verified evidence to stop the ReAct loop.",
                    ready_for_response=True,
                    requires_replan=False,
                )

        max_iterations = state.loop_state.max_iterations or state.max_steps
        if self._criterion_enabled(state, "max_steps_reached") and state.step_index >= max_iterations:
            return StopDecision(
                should_stop=True,
                trigger="max_steps_reached",
                reason=f"Reached the configured maximum of {max_iterations} iteration(s).",
                ready_for_response=True,
                requires_replan=False,
            )

        if state.needs_replan or state.needs_retry or (state.execution.requires_replan if state.execution else False):
            retry_budget = max(0, state.loop_state.retry_budget)
            if self._criterion_enabled(state, "retry_budget_exhausted") and state.retry_count >= retry_budget:
                return StopDecision(
                    should_stop=True,
                    trigger="retry_budget_exhausted",
                    reason=(
                        f"Retry budget exhausted after {state.retry_count} "
                        f"{'retry' if state.retry_count == 1 else 'retries'} "
                        f"with a budget of {retry_budget}."
                    ),
                    ready_for_response=True,
                    requires_replan=False,
                )
            return StopDecision(
                should_stop=False,
                trigger="replan",
                reason=(
                    state.reflection.retry_reason
                    if state.reflection and state.reflection.retry_reason
                    else state.reasoning.replan_reason
                    if state.reasoning and state.reasoning.replan_reason
                    else state.tool_selection.replan_reason
                    if state.tool_selection and state.tool_selection.replan_reason
                    else "The loop must retry with updated reasoning, tools, or constraints."
                ),
                ready_for_response=False,
                requires_replan=True,
            )

        recent_traces = state.react_trace[-2:]
        if (
            self._criterion_enabled(state, "stalled")
            and len(recent_traces) == 2
            and all(not trace.observed_evidence for trace in recent_traces)
        ):
            return StopDecision(
                should_stop=True,
                trigger="stalled",
                reason="Two consecutive iterations failed to produce new observable evidence.",
                ready_for_response=True,
                requires_replan=False,
            )

        return StopDecision(
            should_stop=False,
            trigger="continue",
            reason="Continue the ReAct loop with the next iteration.",
            ready_for_response=False,
            requires_replan=False,
        )

    def _criterion_enabled(self, state: AgentState, trigger: str) -> bool:
        criteria = state.loop_state.stop_conditions
        return not criteria or trigger in criteria
