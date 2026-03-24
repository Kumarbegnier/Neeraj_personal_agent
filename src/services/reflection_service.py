from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.schemas.reflection import ReflectionOutput
from src.schemas.routing import ModelTaskType

from .structured_outputs import estimate_response_completeness


class ReflectionService:
    """Critiques major outputs and can request a single retry with concrete repair guidance."""

    def __init__(self, llm_service: "LLMService") -> None:
        self._llm_service = llm_service

    def critique_output(
        self,
        *,
        stage: str,
        objective: str,
        candidate_output: BaseModel | str,
        success_criteria: list[str] | None = None,
        fallback_output: ReflectionOutput | None = None,
        selected_model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        rendered_output = (
            candidate_output.model_dump_json()
            if isinstance(candidate_output, BaseModel)
            else str(candidate_output)
        )
        fallback = fallback_output or self._fallback_reflection(
            stage=stage,
            candidate_output=candidate_output,
            success_criteria=success_criteria or [],
        )
        return self._llm_service.generate_structured(
            task_type=ModelTaskType.REFLECTION,
            stage=f"{stage}_critique",
            output_type=ReflectionOutput,
            system_prompt=(
                "You are the critique and reflection model for a research-grade multi-model agent system. "
                "Critique the candidate output, identify material issues, and only request a retry if the output "
                "needs a meaningful revision."
            ),
            user_prompt="\n".join(
                [
                    f"Stage: {stage}",
                    f"Objective: {objective}",
                    f"Success criteria: {success_criteria or []}",
                    f"Candidate output: {rendered_output}",
                    f"Fallback critique: {fallback.model_dump_json()}",
                ]
            ),
            fallback_output=fallback,
            selected_model=selected_model,
            metadata=metadata or {},
        )

    def retry_instruction(self, critique: ReflectionOutput) -> str:
        repairs = "; ".join(critique.repairs[:3]) or "Address the critique before retrying."
        return (
            "Reflection requested one retry. Revise the output to address these points: "
            f"{repairs}"
        )

    def _fallback_reflection(
        self,
        *,
        stage: str,
        candidate_output: BaseModel | str,
        success_criteria: list[str],
    ) -> ReflectionOutput:
        completeness = (
            estimate_response_completeness(candidate_output)
            if isinstance(candidate_output, BaseModel)
            else 1.0
            if str(candidate_output).strip()
            else 0.0
        )
        retry_recommended = completeness < 0.45
        issues = []
        repairs = []
        if retry_recommended:
            issues.append(f"The {stage} output looks incomplete.")
            repairs.append("Fill the missing structured fields before finalizing the output.")
        if not success_criteria:
            issues.append("No explicit success criteria were provided for critique.")

        summary = (
            f"Reflection flagged the {stage} output for one retry."
            if retry_recommended
            else f"Reflection accepted the {stage} output."
        )
        return ReflectionOutput(
            summary=summary,
            issues=issues,
            repairs=repairs,
            lessons=["Major outputs should be critiqued before final acceptance."],
            retry_recommended=retry_recommended,
            retry_reason="The candidate output looks materially incomplete." if retry_recommended else None,
            confidence=completeness if completeness > 0 else 0.5,
        )
