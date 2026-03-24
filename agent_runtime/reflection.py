from __future__ import annotations

import json

from src.services.llm_service import LLMService
from src.services.modeling.types import ModelTaskType

from .models import AgentState, ReflectionReport
from .runtime_utils import dedupe_preserve_order, record_model_result, selected_model_override


class ReflectionEngine:
    def __init__(self, llm_service: LLMService | None = None) -> None:
        self._llm_service = llm_service

    def review(self, state: AgentState) -> ReflectionReport:
        fallback = self._fallback_report(state)
        if self._llm_service is None:
            self._apply_to_state(state, fallback)
            return fallback

        model_result = self._llm_service.generate_structured(
            task_type=ModelTaskType.REFLECTION,
            stage="reflection",
            output_type=ReflectionReport,
            system_prompt=(
                "You are the reflection model for a hybrid multi-model runtime. "
                "Translate verification gaps into concrete repairs, route bias, and blocked tool guidance."
            ),
            user_prompt="\n".join(
                [
                    f"Objective: {state.request.message}",
                    f"Verification gaps: {json.dumps(state.verification.gaps if state.verification else [], ensure_ascii=True)}",
                    f"Unverified claims: {json.dumps(state.verification.unverified_claims if state.verification else [], ensure_ascii=True)}",
                    f"Recent tool statuses: {json.dumps([result.status for result in state.last_tool_results], ensure_ascii=True)}",
                    f"Fallback reflection: {fallback.model_dump_json()}",
                ]
            ),
            fallback_output=fallback,
            selected_model=selected_model_override(state),
            metadata={"step_index": state.step_index},
        )
        record_model_result(state, model_result)
        report = ReflectionReport.model_validate(model_result.output.model_dump())
        self._apply_to_state(state, report)
        return report

    def _fallback_report(self, state: AgentState) -> ReflectionReport:
        if state.execution is None or state.verification is None:
            raise ValueError("Execution and verification must exist before reflection.")

        issues = list(state.verification.gaps)
        repairs: list[str] = []
        lessons: list[str] = []
        next_actions: list[str] = []
        blocked_tools: list[str] = []
        route_bias = state.route.agent_name if state.route else None

        if state.verification.unverified_claims:
            issues.append("Some claims still lack strong supporting evidence.")
            repairs.append("Restrict the next attempt to evidence-producing tools and verified observations.")
            if state.route and state.route.agent_name == "general":
                route_bias = "research"
            elif state.route and state.route.agent_name == "communication":
                route_bias = "general"

        gated_tools = [result.tool_name for result in state.last_tool_results if result.status == "gated"]
        if gated_tools:
            issues.append(f"Approval gates blocked: {', '.join(gated_tools)}.")
            repairs.append("Avoid gated tools in the next attempt or switch to a non-side-effect path.")
            blocked_tools.extend(gated_tools)
            if state.route and state.route.agent_name != "general":
                route_bias = "general"

        error_tools = [result.tool_name for result in state.last_tool_results if result.status in {"error", "unavailable"}]
        if error_tools:
            issues.append(f"Unavailable or erroring tools: {', '.join(error_tools)}.")
            repairs.append("Exclude failing tools from the next action set.")
            blocked_tools.extend(error_tools)

        if not issues:
            lessons.append("Verification and reflection formed a closed loop instead of a post-hoc report.")
            lessons.append("The current strategy produced enough evidence to stop the loop.")
            next_actions.append("Persist this successful pattern into memory.")
        else:
            lessons.append("Verification must be able to force a changed strategy, not just annotate failure.")
            next_actions.append("Replan with the new constraints before the next action.")

        checks = [
            f"Observed {len(state.last_tool_results)} tool results at step {state.step_index}.",
            f"Verification status was '{state.verification.status}' with confidence {state.verification.confidence:.2f}.",
            f"Execution confidence was {state.execution.confidence:.2f}.",
        ]

        status = "needs_attention" if issues else "passed"
        summary = (
            "Reflection wrote repair signals back into AgentState."
            if issues
            else "Reflection confirmed the loop can terminate."
        )

        report = ReflectionReport(
            status=status,
            summary=summary,
            checks=checks,
            issues=issues,
            repairs=repairs,
            lessons=lessons,
            next_actions=next_actions,
            retry_recommended=bool(issues),
            retry_reason="Verification surfaced material issues that should trigger one retry." if issues else None,
            confidence=min(state.verification.confidence, state.execution.confidence),
            route_bias=route_bias,
            blocked_tools=dedupe_preserve_order(blocked_tools),
        )
        return report

    def _apply_to_state(self, state: AgentState, report: ReflectionReport) -> None:
        for lesson in report.lessons:
            if lesson not in state.adaptive_constraints and "successful pattern" not in lesson.lower():
                state.adaptive_constraints.append(lesson)

        for repair in report.repairs:
            if repair not in state.adaptive_constraints:
                state.adaptive_constraints.append(repair)

        for tool_name in report.blocked_tools:
            if tool_name not in state.blocked_tools:
                state.blocked_tools.append(tool_name)

        if report.status == "needs_attention":
            state.needs_replan = True
            state.needs_retry = True
            if report.route_bias:
                state.route_bias = report.route_bias
        else:
            state.needs_replan = False
            state.needs_retry = False
