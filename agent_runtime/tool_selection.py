from __future__ import annotations

import json

from src.services.llm_service import LLMService
from src.services.modeling.types import ModelTaskType

from .models import AgentState, ToolRequest, ToolSelectionDecision
from .runtime_utils import dedupe_preserve_order, record_model_result, selected_model_override


class ToolSelectionEngine:
    def __init__(self, llm_service: LLMService | None = None) -> None:
        self._llm_service = llm_service

    def select(self, state: AgentState) -> tuple[ToolSelectionDecision, list[ToolRequest]]:
        fallback_selection, fallback_requests = self._fallback_selection(state)
        if self._llm_service is None:
            return fallback_selection, fallback_requests

        model_result = self._llm_service.generate_structured(
            task_type=ModelTaskType.REASONING,
            stage="tool_selection",
            output_type=ToolSelectionDecision,
            system_prompt=(
                "You are the tool selection layer inside a research-grade ReAct agent runtime. "
                "Choose the smallest useful set of tools for the next action, defer lower-value tools, "
                "and request replanning if the current candidate set cannot make progress."
            ),
            user_prompt=self._tool_selection_prompt(state, fallback_selection),
            fallback_output=fallback_selection,
            selected_model=selected_model_override(state),
            metadata={"step_index": state.step_index, "route": state.route.agent_name if state.route else "general"},
        )
        record_model_result(state, model_result)
        selection = ToolSelectionDecision.model_validate(model_result.output.model_dump())
        selected_requests = self._apply_selection(state.decision.tool_requests if state.decision else [], selection)
        if not selected_requests:
            selection = fallback_selection
            selected_requests = fallback_requests
        return selection, selected_requests

    def _fallback_selection(self, state: AgentState) -> tuple[ToolSelectionDecision, list[ToolRequest]]:
        if state.decision is None:
            raise ValueError("Agent decision must exist before tool selection.")

        candidate_requests = self._candidate_requests(state)
        preferred_order = dedupe_preserve_order(
            [
                *(state.reasoning.candidate_tools if state.reasoning else []),
                *(request.tool_name for request in candidate_requests),
            ]
        )
        prioritized = sorted(
            candidate_requests,
            key=lambda request: (
                preferred_order.index(request.tool_name) if request.tool_name in preferred_order else len(preferred_order),
                request.priority,
            ),
        )
        selected_requests = prioritized[: min(3, len(prioritized))]
        selected_names = [request.tool_name for request in selected_requests]
        deferred_names = [request.tool_name for request in prioritized if request.tool_name not in selected_names]
        requires_replan = not selected_requests or bool(state.reasoning.should_replan if state.reasoning else False)
        selection = ToolSelectionDecision(
            selected_tool_names=selected_names,
            deferred_tool_names=deferred_names,
            rationale=(
                "Prioritize the highest-signal, non-blocked tools that align with the current reasoning step."
            ),
            expected_outcome=(
                state.reasoning.expected_observation
                if state.reasoning and state.reasoning.expected_observation
                else "Produce fresh evidence for the current objective."
            ),
            requires_replan=requires_replan,
            replan_reason=(
                state.reasoning.replan_reason
                if state.reasoning and state.reasoning.replan_reason
                else "No viable tools were available for the current step."
                if requires_replan
                else None
            ),
        )
        return selection, selected_requests

    def _candidate_requests(self, state: AgentState) -> list[ToolRequest]:
        blocked = {tool_name.lower() for tool_name in state.blocked_tools}
        requests = [
            request.model_copy(deep=True)
            for request in (state.decision.tool_requests if state.decision else [])
            if request.tool_name.lower() not in blocked
        ]
        deduped: list[ToolRequest] = []
        seen: set[str] = set()
        for request in sorted(requests, key=lambda item: item.priority):
            if request.tool_name in seen:
                continue
            deduped.append(request)
            seen.add(request.tool_name)
        return deduped

    def _apply_selection(
        self,
        requests: list[ToolRequest],
        selection: ToolSelectionDecision,
    ) -> list[ToolRequest]:
        selected = {name for name in selection.selected_tool_names}
        chosen = [request.model_copy(deep=True) for request in requests if request.tool_name in selected]
        return sorted(chosen, key=lambda request: request.priority)

    def _tool_selection_prompt(self, state: AgentState, fallback: ToolSelectionDecision) -> str:
        return "\n".join(
            [
                f"Objective: {state.request.message}",
                f"Architecture mode: {state.architecture.mode.value if state.architecture else ''}",
                f"Reasoning summary: {state.reasoning.reasoning_summary if state.reasoning else ''}",
                f"Action strategy: {state.reasoning.action_strategy if state.reasoning else ''}",
                f"Plan required tools: {json.dumps(state.plan.required_tools if state.plan else [], ensure_ascii=True)}",
                f"Blocked tools: {json.dumps(state.blocked_tools, ensure_ascii=True)}",
                f"Candidate tool requests: {json.dumps([request.model_dump(mode='json') for request in self._candidate_requests(state)], ensure_ascii=True, default=str)}",
                f"Fallback tool selection: {fallback.model_dump_json()}",
            ]
        )
