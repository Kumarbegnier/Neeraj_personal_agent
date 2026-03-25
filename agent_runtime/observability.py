from __future__ import annotations

from typing import Iterable

from src.schemas.observability import AutonomyMetrics, RuntimeTrace, StepTrace

from .models import AgentState, ToolResult


class RuntimeObservabilityEngine:
    """Builds typed runtime observability artifacts from loop state."""

    _IRREVERSIBLE_SIDE_EFFECTS = frozenset({"delete", "deploy", "purchase", "send", "shutdown"})
    _TOOL_FAILURE_STATUSES = frozenset(
        {
            "blocked",
            "error",
            "failed",
            "gated",
            "invalid_input",
            "invalid_output",
            "verification_failed",
        }
    )
    _RECOVERY_STATUSES = frozenset({"completed", "goal_achieved", "ready", "verified"})
    _RECOVERY_VERIFICATION_STATUSES = frozenset({"passed", "skipped"})

    def record_iteration(self, state: AgentState) -> StepTrace:
        step_trace = self._build_step_trace(state, phase=state.loop_state.phase.value)
        if state.step_traces and self._step_needs_recovery(state.step_traces[-1]) and self._step_recovered(step_trace):
            step_trace = step_trace.model_copy(update={"recovered_after_failure": True})

        state.step_traces.append(step_trace)
        pending_retry = (
            1
            if state.stop_decision is not None and state.stop_decision.requires_replan and not state.stop_decision.should_stop
            else 0
        )
        state.autonomy_metrics = self._aggregate_metrics(
            state.step_traces,
            retry_count=state.retry_count + pending_retry,
        )
        return step_trace

    def record_preflight_block(self, state: AgentState) -> StepTrace:
        step_trace = self._build_step_trace(state, phase="preflight")
        state.step_traces.append(step_trace)
        state.autonomy_metrics = self._aggregate_metrics(state.step_traces, retry_count=state.retry_count)
        return step_trace

    def finalize_trace(self, state: AgentState) -> RuntimeTrace:
        state.autonomy_metrics = self._aggregate_metrics(state.step_traces, retry_count=state.retry_count)
        state.runtime_trace = RuntimeTrace(
            request_id=state.gateway.request_id if state.gateway is not None else "",
            state_id=state.state_id,
            user_id=state.request.user_id,
            session_id=state.request.session_id,
            objective=state.plan.objective if state.plan is not None else state.request.message.strip(),
            assigned_agent=state.route.agent_name if state.route is not None else "general",
            architecture_mode=state.architecture.mode.value if state.architecture is not None else "",
            termination_reason=state.termination_reason,
            summary=self._runtime_summary(state),
            autonomy_metrics=state.autonomy_metrics.model_copy(deep=True),
            steps=[step.model_copy(deep=True) for step in state.step_traces],
            metadata={
                "approval_mode": state.session.permission.mode.value if state.session is not None else "unknown",
                "approval_required": (
                    state.safety.permission.requires_confirmation
                    if state.safety is not None
                    else state.session.permission.requires_confirmation
                    if state.session is not None
                    else False
                ),
                "loop_count": state.step_index,
                "trace_event_count": len(state.trace),
                "tool_result_count": len(state.tool_history),
                "model_run_count": len(state.model_runs),
                "handoff_available": state.handoff_packet is not None,
                "last_handoff_id": state.handoff_packet.handoff_id if state.handoff_packet is not None else None,
            },
        )
        return state.runtime_trace

    def _build_step_trace(self, state: AgentState, *, phase: str) -> StepTrace:
        selected_tools = self._selected_tools(state)
        tool_statuses = self._tool_statuses(state.last_tool_results)
        approvals_requested = self._approvals_requested(state)
        human_intervention_required = approvals_requested > 0 or (
            bool(state.request.metadata.get("approval_granted")) and state.step_index <= 1
        )
        step_status = self._step_status(state, approvals_requested=approvals_requested, phase=phase)
        return StepTrace(
            step_index=state.step_index,
            agent_name=state.route.agent_name if state.route is not None else "general",
            architecture_mode=state.architecture.mode.value if state.architecture is not None else "",
            phase=phase,
            status=step_status,
            summary=self._step_summary(state, approvals_requested=approvals_requested, phase=phase),
            selected_tools=selected_tools,
            tool_statuses=tool_statuses,
            autonomous=not human_intervention_required,
            approvals_requested=approvals_requested,
            retry_triggered=self._retry_triggered(state),
            human_intervention_required=human_intervention_required,
            irreversible_actions_attempted=self._irreversible_actions_attempted(state.last_tool_results),
            verification_status=state.verification.status if state.verification is not None else "pending",
            reflection_status=state.reflection.status if state.reflection is not None else "pending",
            termination_signal=state.stop_decision.trigger if state.stop_decision is not None else "continue",
            metadata={
                "approval_granted": bool(state.request.metadata.get("approval_granted")),
                "blocked_tools": list(state.blocked_tools),
                "retry_count": state.retry_count,
                "replan_count": state.replan_count,
                "trace_event_count": len(state.trace),
                "stop_reason": state.stop_decision.reason if state.stop_decision is not None else "",
            },
        )

    def _aggregate_metrics(
        self,
        step_traces: Iterable[StepTrace],
        *,
        retry_count: int,
    ) -> AutonomyMetrics:
        steps = list(step_traces)
        total_steps = len(steps)
        human_intervention_events = sum(1 for step in steps if step.human_intervention_required)
        return AutonomyMetrics(
            total_steps=total_steps,
            autonomous_steps_count=sum(1 for step in steps if step.autonomous),
            approvals_requested=sum(step.approvals_requested for step in steps),
            retries_used=retry_count,
            recovery_count_after_failure=sum(1 for step in steps if step.recovered_after_failure),
            human_intervention_events=human_intervention_events,
            human_intervention_ratio=round(human_intervention_events / total_steps, 2) if total_steps else 0.0,
            irreversible_actions_attempted=sum(step.irreversible_actions_attempted for step in steps),
        )

    def _selected_tools(self, state: AgentState) -> list[str]:
        if state.tool_selection is not None and state.tool_selection.selected_tool_names:
            return list(state.tool_selection.selected_tool_names)
        if state.last_tool_results:
            return [result.tool_name for result in state.last_tool_results]
        if state.pending_tool_requests:
            return [request.tool_name for request in state.pending_tool_requests]
        return []

    def _tool_statuses(self, tool_results: list[ToolResult]) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for result in tool_results:
            key = result.tool_name if result.tool_name not in statuses else f"{result.tool_name}:{result.call_id or len(statuses)}"
            statuses[key] = result.status
        return statuses

    def _approvals_requested(self, state: AgentState) -> int:
        tool_level_requests = sum(1 for result in state.last_tool_results if self._tool_requires_approval(result))
        session_level_request = (
            state.step_index <= 1
            and state.session is not None
            and state.session.permission.requires_confirmation
            and not bool(state.request.metadata.get("approval_granted"))
            and tool_level_requests == 0
        )
        return tool_level_requests + (1 if session_level_request else 0)

    def _tool_requires_approval(self, result: ToolResult) -> bool:
        return (
            result.status == "gated"
            or result.blocked_reason == "confirmation_required"
            or bool(result.output.get("requires_confirmation"))
            or bool(result.output.get("approval_required"))
        )

    def _irreversible_actions_attempted(self, tool_results: list[ToolResult]) -> int:
        return sum(1 for result in tool_results if result.side_effect in self._IRREVERSIBLE_SIDE_EFFECTS)

    def _retry_triggered(self, state: AgentState) -> bool:
        return bool(
            state.needs_retry
            or state.needs_replan
            or (state.execution.requires_replan if state.execution is not None else False)
            or (state.verification.retry_recommended if state.verification is not None else False)
            or (state.stop_decision.requires_replan if state.stop_decision is not None else False)
        )

    def _step_status(
        self,
        state: AgentState,
        *,
        approvals_requested: int,
        phase: str,
    ) -> str:
        if phase == "preflight":
            if state.session is not None and state.session.permission.mode.value == "blocked":
                return "blocked"
            if state.gateway is not None and not state.gateway.accepted:
                return "blocked"
            if approvals_requested > 0:
                return "needs_approval"
            return "preflight"

        if approvals_requested > 0:
            return "needs_approval"
        if state.stop_decision is not None and state.stop_decision.should_stop:
            return state.stop_decision.trigger
        if state.execution is None and self._retry_triggered(state):
            return "replanned"
        if any(result.status in self._TOOL_FAILURE_STATUSES for result in state.last_tool_results):
            return "failed"
        if state.execution is not None and state.execution.ready_for_response:
            return "ready"
        if state.verification is not None and state.verification.status == "passed":
            return "verified"
        return state.goal_status or state.status or "in_progress"

    def _step_summary(
        self,
        state: AgentState,
        *,
        approvals_requested: int,
        phase: str,
    ) -> str:
        if phase == "preflight":
            return state.loop_state.last_stop_reason or "Execution was blocked before the loop started."
        if approvals_requested > 0:
            return (
                "Execution paused for approval-gated actions."
                if state.last_tool_results
                else "Execution requires explicit approval before side effects can proceed."
            )
        for candidate in (
            state.stop_decision.reason if state.stop_decision is not None else "",
            state.execution.summary if state.execution is not None else "",
            state.verification.summary if state.verification is not None else "",
            state.reflection.summary if state.reflection is not None else "",
        ):
            if candidate:
                return candidate
        return "The runtime completed the iteration without a detailed observability summary."

    def _step_needs_recovery(self, step_trace: StepTrace) -> bool:
        return bool(
            step_trace.approvals_requested
            or step_trace.retry_triggered
            or step_trace.human_intervention_required
            or step_trace.status in {"blocked", "failed", "needs_approval", "replanned", "retry_budget_exhausted", "stalled"}
            or step_trace.verification_status not in self._RECOVERY_VERIFICATION_STATUSES
        )

    def _step_recovered(self, step_trace: StepTrace) -> bool:
        return (
            step_trace.status in self._RECOVERY_STATUSES
            or (
                not step_trace.approvals_requested
                and not step_trace.retry_triggered
                and step_trace.verification_status in self._RECOVERY_VERIFICATION_STATUSES
            )
        )

    def _runtime_summary(self, state: AgentState) -> str:
        if state.final_response.strip():
            return state.final_response[:280]
        if state.verification is not None and state.verification.summary:
            return state.verification.summary
        if state.stop_decision is not None and state.stop_decision.reason:
            return state.stop_decision.reason
        return "Runtime observability captured the latest execution outcome."
