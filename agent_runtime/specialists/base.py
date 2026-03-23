from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AgentDecision, AgentState, ExecutionResult, SkillDescriptor, ToolResult
from .common import memory_text


class BaseAgent(ABC):
    name = "base"

    @abstractmethod
    def decide(self, state: AgentState, skills: list[SkillDescriptor]) -> AgentDecision:
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
