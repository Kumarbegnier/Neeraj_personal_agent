from __future__ import annotations

import json
from typing import Any

from src.schemas.planner import PlannerOutput
from src.schemas.reflection import ReflectionOutput
from src.schemas.routing import ModelTaskType

from .reflection_service import ReflectionService


class PlannerService:
    """Generates structured planner output and allows one critique-driven retry."""

    def __init__(
        self,
        llm_service: "LLMService",
        reflection_service: ReflectionService | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._reflection_service = reflection_service or ReflectionService(llm_service)

    def generate_plan(
        self,
        *,
        objective: str,
        intent: str,
        complexity: str,
        requested_capabilities: list[str],
        constraints: list[str],
        verification_focus: list[str],
        fallback_output: PlannerOutput,
        selected_model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        prompt = self._prompt(
            objective=objective,
            intent=intent,
            complexity=complexity,
            requested_capabilities=requested_capabilities,
            constraints=constraints,
            verification_focus=verification_focus,
            fallback_output=fallback_output,
            critique_instruction=None,
        )
        result = self._llm_service.generate_structured(
            task_type=ModelTaskType.PLANNING,
            stage="planning",
            output_type=PlannerOutput,
            system_prompt=(
                "You are the structured planning model for a research-grade multi-model agent runtime. "
                "Return a planner output with a crisp task summary, concrete subtasks, required tools, risk level, "
                "approval need, and success criteria."
            ),
            user_prompt=prompt,
            fallback_output=fallback_output,
            selected_model=selected_model,
            metadata=metadata or {},
        )
        critique = self._reflection_service.critique_output(
            stage="planning",
            objective=objective,
            candidate_output=result.output,
            success_criteria=fallback_output.success_criteria,
            fallback_output=ReflectionOutput(
                summary="Planner critique completed.",
                issues=[],
                repairs=[],
                lessons=["Planning output should be complete and executable."],
                retry_recommended=False,
                confidence=0.8,
            ),
            selected_model=selected_model,
            metadata=metadata or {},
        )
        if critique.output.retry_recommended:
            retry_prompt = self._prompt(
                objective=objective,
                intent=intent,
                complexity=complexity,
                requested_capabilities=requested_capabilities,
                constraints=constraints,
                verification_focus=verification_focus,
                fallback_output=fallback_output,
                critique_instruction=self._reflection_service.retry_instruction(critique.output),
            )
            result = self._llm_service.generate_structured(
                task_type=ModelTaskType.PLANNING,
                stage="planning_retry",
                output_type=PlannerOutput,
                system_prompt=(
                    "You are the structured planning model for a research-grade multi-model agent runtime. "
                    "Return a revised planner output that explicitly addresses the critique."
                ),
                user_prompt=retry_prompt,
                fallback_output=fallback_output,
                selected_model=selected_model,
                metadata={**(metadata or {}), "retry_requested": True},
            )
        return result, critique

    def _prompt(
        self,
        *,
        objective: str,
        intent: str,
        complexity: str,
        requested_capabilities: list[str],
        constraints: list[str],
        verification_focus: list[str],
        fallback_output: PlannerOutput,
        critique_instruction: str | None,
    ) -> str:
        parts = [
            f"Objective: {objective}",
            f"Intent: {intent}",
            f"Complexity: {complexity}",
            f"Requested capabilities: {json.dumps(requested_capabilities, ensure_ascii=True)}",
            f"Constraints: {json.dumps(constraints, ensure_ascii=True)}",
            f"Verification focus: {json.dumps(verification_focus, ensure_ascii=True)}",
            f"Fallback planner output: {fallback_output.model_dump_json()}",
        ]
        if critique_instruction:
            parts.append(f"Critique instruction: {critique_instruction}")
        return "\n".join(parts)
