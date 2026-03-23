from __future__ import annotations

from .agents import BaseAgent
from .models import AgentState, ExecutionResult
from .tools import ToolLayer


class ExecutionEngine:
    def __init__(self, tool_layer: ToolLayer) -> None:
        self._tool_layer = tool_layer

    def execute(self, state: AgentState, agent: BaseAgent) -> ExecutionResult:
        if state.decision is None:
            raise ValueError("Agent decision must exist before execution.")
        if state.context is None:
            raise ValueError("Context must exist before execution.")

        tool_results = self._tool_layer.run_many(state.pending_tool_requests, state.context)
        result = agent.assess(state, state.decision, tool_results)
        return result
