from __future__ import annotations

import json

from src.services.llm_service import LLMService
from src.services.modeling.types import ModelTaskType

from .agents import BaseAgent
from .models import AgentState, ExecutionResult
from .tools import ToolLayer
from .runtime_utils import record_model_result, selected_model_override


class ExecutionEngine:
    def __init__(self, tool_layer: ToolLayer, llm_service: LLMService | None = None) -> None:
        self._tool_layer = tool_layer
        self._llm_service = llm_service

    def execute(self, state: AgentState, agent: BaseAgent) -> ExecutionResult:
        if state.decision is None:
            raise ValueError("Agent decision must exist before execution.")
        if state.context is None:
            raise ValueError("Context must exist before execution.")

        tool_results = self._tool_layer.run_many(state.pending_tool_requests, state.context)
        fallback = agent.assess(state, state.decision, tool_results)
        if self._llm_service is None:
            return fallback

        model_result = self._llm_service.generate_structured(
            task_type=ModelTaskType.TOOL_EXECUTION,
            stage="execution",
            output_type=ExecutionResult,
            system_prompt=(
                "You are the execution summarizer for a hybrid multi-model agent runtime. "
                "Turn tool results into a structured execution result without inventing unsupported evidence."
            ),
            user_prompt="\n".join(
                [
                    f"Agent name: {agent.name}",
                    f"Decision summary: {state.decision.summary}",
                    f"Tool results: {json.dumps([result.model_dump(mode='json') for result in tool_results], ensure_ascii=True, default=str)}",
                    f"Fallback execution result: {fallback.model_dump_json()}",
                ]
            ),
            fallback_output=fallback,
            selected_model=selected_model_override(state),
            metadata={"step_index": state.step_index, "agent": agent.name},
        )
        record_model_result(state, model_result)
        return ExecutionResult.model_validate(model_result.output.model_dump())
