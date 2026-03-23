from __future__ import annotations

import json
from abc import ABC, abstractmethod

from src.services.llm_service import LLMService
from src.services.modeling.types import ModelTaskType

from ..models import AgentDecision, AgentState, ExecutionResult, SkillDescriptor, ToolResult
from ..runtime_utils import record_model_result, selected_model_override
from .common import memory_text


class BaseAgent(ABC):
    name = "base"
    decision_task_type = ModelTaskType.REASONING

    def __init__(self, llm_service: LLMService | None = None) -> None:
        self._llm_service = llm_service

    def decide(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
        fallback = self.build_decision(state, skills)
        if self._llm_service is None:
            return fallback

        model_result = self._llm_service.generate_structured(
            task_type=self.decision_task_type,
            stage=f"{self.name}_decision",
            output_type=AgentDecision,
            system_prompt=(
                "You are a specialist decision model inside a hybrid multi-model runtime. "
                "Return a structured agent decision that stays aligned with the current specialist lane."
            ),
            user_prompt="\n".join(
                [
                    f"Specialist: {self.name}",
                    f"User message: {state.request.message}",
                    f"Available skills: {json.dumps([skill.name for skill in skills], ensure_ascii=True)}",
                    f"Blocked tools: {json.dumps(state.blocked_tools, ensure_ascii=True)}",
                    f"Memory summary: {memory_text(state)}",
                    f"Fallback decision: {fallback.model_dump_json()}",
                ]
            ),
            fallback_output=fallback,
            selected_model=selected_model_override(state),
            metadata={"step_index": state.step_index, "agent": self.name},
        )
        record_model_result(state, model_result)
        return AgentDecision.model_validate(model_result.output.model_dump())

    @abstractmethod
    def build_decision(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
        raise NotImplementedError

    @abstractmethod
    def assess(
        self,
        state: AgentState,
        decision: AgentDecision,
        tool_results: list[ToolResult],
    ) -> ExecutionResult:
        raise NotImplementedError

    def _memory_driven_verification(self, state: AgentState) -> bool:
        text = memory_text(state)
        return state.retry_count > 0 or "verification" in text or "reflection lesson" in text
