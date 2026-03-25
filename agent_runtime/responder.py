from __future__ import annotations

from src.schemas.claims import ClaimVerificationReport
from src.schemas.reflection import ReflectionOutput
from src.services.reflection_service import ReflectionService
from src.services.llm_service import LLMService
from src.services.modeling.types import ModelTaskType

from .claim_verifier import ClaimVerifier
from .models import AgentState, StructuredResponse
from .runtime_utils import record_model_result, selected_model_override


class ResponseComposer:
    def __init__(
        self,
        llm_service: LLMService | None = None,
        reflection_service: ReflectionService | None = None,
        claim_verifier: ClaimVerifier | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._reflection_service = reflection_service or (
            ReflectionService(llm_service) if llm_service is not None else None
        )
        self._claim_verifier = claim_verifier or ClaimVerifier()

    def compose(self, state: AgentState) -> str:
        return self.synthesize_from_state(state)

    def synthesize_from_state(self, state: AgentState) -> str:
        fallback = self._fallback_response(state)
        fallback_claim_report = self._verify_candidate(state, fallback.response, source="fallback_response")
        fallback = self._apply_claim_verification(fallback, fallback_claim_report)
        if self._llm_service is None:
            state.claim_verification = fallback_claim_report
            return fallback.response

        base_prompt = self._response_prompt(state, fallback, claim_report=fallback_claim_report)
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
        final_claim_report = self._verify_candidate(state, structured.response, source="final_response")
        structured = self._apply_claim_verification(structured, final_claim_report)

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
            claim_retry_instruction = self._claim_retry_instruction(final_claim_report)
            if critique.retry_recommended or claim_retry_instruction:
                critique_parts: list[str] = []
                if critique.retry_recommended:
                    critique_parts.append(self._reflection_service.retry_instruction(critique))
                if claim_retry_instruction:
                    critique_parts.append(claim_retry_instruction)
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
                        claim_report=final_claim_report,
                        critique_instruction=" ".join(critique_parts).strip(),
                    ),
                    fallback_output=fallback,
                    selected_model=selected_model_override(state),
                    metadata={"step_index": state.step_index, "retry_requested": True},
                )
                record_model_result(state, retry_result)
                structured = StructuredResponse.model_validate(retry_result.output.model_dump())
                final_claim_report = self._verify_candidate(state, structured.response, source="response_retry")
                structured = self._apply_claim_verification(structured, final_claim_report)
        state.claim_verification = final_claim_report
        return structured.response

    def _fallback_response(self, state: AgentState) -> StructuredResponse:
        agent_name = state.route.agent_name if state.route else "general"
        execution_summary = state.execution.summary if state.execution else "No execution summary was produced."
        observation_clause = self._observation_clause(state)
        verification_clause = self._verification_clause(state)
        claim_clause = self._claim_clause(state)
        reflection_clause = self._reflection_clause(state)
        safety_clause = self._safety_clause(state)

        response = " ".join(
            part
            for part in [
                f"The {agent_name} agent completed the request after {state.step_index} loop step(s).",
                execution_summary,
                observation_clause,
                verification_clause,
                claim_clause,
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
                claim_clause,
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

    def _claim_clause(self, state: AgentState) -> str:
        claim_report = state.claim_verification or (
            state.verification.claim_verification if state.verification else None
        )
        if not claim_report or not claim_report.enabled:
            return ""
        if claim_report.status == "passed":
            return "Claim verification found the final content strongly grounded."
        return f"Claim verification still needs attention: {claim_report.confidence_summary}"

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
        claim_report: ClaimVerificationReport | None = None,
        critique_instruction: str | None = None,
    ) -> str:
        parts = [
            f"Assigned agent: {state.route.agent_name if state.route else 'general'}",
            f"Execution summary: {state.execution.summary if state.execution else ''}",
            f"Verification summary: {state.verification.summary if state.verification else ''}",
            f"Claim verification summary: {state.claim_verification.summary if state.claim_verification else ''}",
            f"Reflection summary: {state.reflection.summary if state.reflection else ''}",
            f"Safety status: {state.safety.status if state.safety else 'unknown'}",
            f"Fallback response: {fallback.model_dump_json()}",
        ]
        if claim_report is not None:
            parts.append(f"Candidate claim verification: {claim_report.model_dump_json()}")
        if critique_instruction:
            parts.append(f"Critique instruction: {critique_instruction}")
        return "\n".join(parts)

    def _verify_candidate(
        self,
        state: AgentState,
        candidate_response: str,
        *,
        source: str,
    ) -> ClaimVerificationReport:
        return self._claim_verifier.verify(
            state,
            candidate_response=candidate_response,
            source=source,
        )

    def _apply_claim_verification(
        self,
        structured: StructuredResponse,
        claim_report: ClaimVerificationReport,
    ) -> StructuredResponse:
        if not claim_report.enabled:
            return structured

        response = structured.response
        highlights = [highlight for highlight in structured.highlights if highlight]
        note = claim_report.confidence_summary or claim_report.summary
        caution = self._claim_caution_clause(claim_report)
        if caution and caution not in response:
            response = f"{response} {caution}".strip()
        if note and note not in highlights:
            highlights.append(note)

        return structured.model_copy(
            update={
                "response": response,
                "highlights": highlights[:4],
                "claim_verification_note": note,
            }
        )

    def _claim_retry_instruction(self, claim_report: ClaimVerificationReport) -> str:
        if not claim_report.enabled:
            return ""
        if not claim_report.unsupported_claims and len(claim_report.weakly_supported_claims) < 2:
            return ""
        risky_claims = claim_report.unsupported_claims or claim_report.weakly_supported_claims
        return (
            "Revise the answer so every substantive claim is either grounded in the recorded evidence "
            f"or explicitly softened with uncertainty. Rework: {'; '.join(risky_claims[:2])}."
        )

    def _claim_caution_clause(self, claim_report: ClaimVerificationReport) -> str:
        if not claim_report.enabled or claim_report.status == "passed":
            return ""
        return f"Caution: {claim_report.confidence_summary}"
