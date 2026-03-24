from __future__ import annotations

from src.schemas.reflection import ReflectionOutput
from src.services.reflection_service import ReflectionService
from src.services.llm_service import LLMService
from src.services.modeling.types import ModelTaskType

from .models import AgentState, StructuredResponse
from .runtime_utils import record_model_result, selected_model_override


class ResponseComposer:
    def __init__(
        self,
        llm_service: LLMService | None = None,
        reflection_service: ReflectionService | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._reflection_service = reflection_service or (
            ReflectionService(llm_service) if llm_service is not None else None
        )

    def compose(self, state: AgentState) -> str:
        return self.synthesize_from_state(state)

    def synthesize_from_state(self, state: AgentState) -> str:
        fallback = self._fallback_response(state)
        if self._llm_service is None:
            return fallback.response

        base_prompt = self._response_prompt(state, fallback)
        model_result = self._llm_service.generate_structured(
            task_type=ModelTaskType.COMMUNICATION,
            stage="response",
            output_type=StructuredResponse,
            system_prompt=(
                "You are the final communication layer for a research-grade multi-model agent system. "
                "Produce a concise final answer grounded in the converged runtime state."
            ),
            user_prompt=base_prompt,
            fallback_output=fallback,
            selected_model=selected_model_override(state),
            metadata={"step_index": state.step_index},
        )
        record_model_result(state, model_result)
        structured = StructuredResponse.model_validate(model_result.output.model_dump())

        if self._reflection_service is not None:
            critique_result = self._reflection_service.critique_output(
                stage="response",
                objective=state.request.message,
                candidate_output=structured,
                success_criteria=state.plan.success_criteria if state.plan else [],
                fallback_output=ReflectionOutput(
                    summary="Response critique completed.",
                    issues=[],
                    repairs=[],
                    lessons=["Major user-facing outputs should be critiqued before final acceptance."],
                    retry_recommended=False,
                    confidence=0.85,
                ),
                selected_model=selected_model_override(state),
                metadata={"step_index": state.step_index},
            )
            record_model_result(state, critique_result)
            critique = ReflectionOutput.model_validate(critique_result.output.model_dump())
            if critique.retry_recommended:
                retry_result = self._llm_service.generate_structured(
                    task_type=ModelTaskType.COMMUNICATION,
                    stage="response_retry",
                    output_type=StructuredResponse,
                    system_prompt=(
                        "You are the final communication layer for a research-grade multi-model agent system. "
                        "Produce a concise revised answer that addresses the critique."
                    ),
                    user_prompt=self._response_prompt(
                        state,
                        fallback,
                        critique_instruction=self._reflection_service.retry_instruction(critique),
                    ),
                    fallback_output=fallback,
                    selected_model=selected_model_override(state),
                    metadata={"step_index": state.step_index, "retry_requested": True},
                )
                record_model_result(state, retry_result)
                structured = StructuredResponse.model_validate(retry_result.output.model_dump())
        return structured.response

    def _fallback_response(self, state: AgentState) -> StructuredResponse:
        agent_name = state.route.agent_name if state.route else "general"
        execution_summary = state.execution.summary if state.execution else "No execution summary was produced."
        observation_clause = self._observation_clause(state)
        verification_clause = self._verification_clause(state)
        reflection_clause = self._reflection_clause(state)
        safety_clause = self._safety_clause(state)

        response = " ".join(
            part
            for part in [
                f"The {agent_name} agent completed the request after {state.step_index} loop step(s).",
                execution_summary,
                observation_clause,
                verification_clause,
                reflection_clause,
                safety_clause,
            ]
            if part
        ).strip()
        return StructuredResponse(
            response=response,
            highlights=[
                execution_summary,
                verification_clause,
                reflection_clause,
            ],
            approval_note=safety_clause,
        )

    def _observation_clause(self, state: AgentState) -> str:
        if not state.execution or not state.execution.observations:
            return ""
        return f"Key observations: {'; '.join(state.execution.observations[:3])}."

    def _verification_clause(self, state: AgentState) -> str:
        if not state.verification:
            return ""
        if state.verification.status == "passed":
            return "Verification passed for the final loop state."
        return f"Verification still has gaps: {'; '.join(state.verification.gaps[:2])}."

    def _reflection_clause(self, state: AgentState) -> str:
        if not state.reflection:
            return ""
        if state.reflection.status == "passed":
            return "Reflection found the current strategy sufficient to stop."
        return f"Reflection forced strategy changes such as {', '.join(state.reflection.repairs[:2])}."

    def _safety_clause(self, state: AgentState) -> str:
        if not state.safety:
            return ""
        if state.safety.permission.requires_confirmation:
            return "Additional confirmation is required before any side effect."
        return "No additional approval gate was triggered."

    def _response_prompt(
        self,
        state: AgentState,
        fallback: StructuredResponse,
        critique_instruction: str | None = None,
    ) -> str:
        parts = [
            f"Assigned agent: {state.route.agent_name if state.route else 'general'}",
            f"Execution summary: {state.execution.summary if state.execution else ''}",
            f"Verification summary: {state.verification.summary if state.verification else ''}",
            f"Reflection summary: {state.reflection.summary if state.reflection else ''}",
            f"Safety status: {state.safety.status if state.safety else 'unknown'}",
            f"Fallback response: {fallback.model_dump_json()}",
        ]
        if critique_instruction:
            parts.append(f"Critique instruction: {critique_instruction}")
        return "\n".join(parts)
