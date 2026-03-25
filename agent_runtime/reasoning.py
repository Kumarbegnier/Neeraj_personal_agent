from __future__ import annotations

import json

from src.schemas.reflection import ReflectionOutput
from src.services.llm_service import LLMService
from src.services.modeling.types import ModelTaskType
from src.services.reflection_service import ReflectionService

from .models import AgentState, ReasoningStep
from .runtime_utils import recent_observation_summaries, record_model_result, selected_model_override


class ReasoningEngine:
    def __init__(
        self,
        llm_service: LLMService | None = None,
        reflection_service: ReflectionService | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._reflection_service = reflection_service or (
            ReflectionService(llm_service) if llm_service is not None else None
        )

    def reason(self, state: AgentState) -> ReasoningStep:
        fallback = self._fallback_reasoning(state)
        if self._llm_service is None:
            return fallback

        model_result = self._llm_service.generate_structured(
            task_type=ModelTaskType.REASONING,
            stage="reasoning",
            output_type=ReasoningStep,
            system_prompt=(
                "You are the reasoning layer inside a research-grade ReAct agent runtime. "
                "Given the current state and candidate tools, produce the next thought and action strategy "
                "without leaking provider-specific behavior."
            ),
            user_prompt=self._reasoning_prompt(state, fallback),
            fallback_output=fallback,
            selected_model=selected_model_override(state),
            metadata={"step_index": state.step_index, "route": state.route.agent_name if state.route else "general"},
        )
        record_model_result(state, model_result)
        reasoning = ReasoningStep.model_validate(model_result.output.model_dump())

        if self._reflection_service is None:
            return reasoning

        critique_result = self._reflection_service.critique_output(
            stage="reasoning",
            objective=state.request.message,
            candidate_output=reasoning,
            success_criteria=state.plan.success_criteria if state.plan else [],
            fallback_output=ReflectionOutput(
                summary="Reasoning critique completed.",
                issues=[],
                repairs=[],
                lessons=["Reasoning should justify the next action with evidence and clear intent."],
                retry_recommended=False,
                confidence=0.8,
            ),
            selected_model=selected_model_override(state),
            metadata={"step_index": state.step_index, "route": state.route.agent_name if state.route else "general"},
        )
        record_model_result(state, critique_result)
        critique = ReflectionOutput.model_validate(critique_result.output.model_dump())
        if critique.retry_recommended:
            retry_result = self._llm_service.generate_structured(
                task_type=ModelTaskType.REASONING,
                stage="reasoning_retry",
                output_type=ReasoningStep,
                system_prompt=(
                    "You are the reasoning layer inside a research-grade ReAct agent runtime. "
                    "Revise the reasoning so it directly addresses the critique and remains actionable."
                ),
                user_prompt="\n".join(
                    [
                        self._reasoning_prompt(state, fallback),
                        f"Critique instruction: {self._reflection_service.retry_instruction(critique)}",
                    ]
                ),
                fallback_output=fallback,
                selected_model=selected_model_override(state),
                metadata={"step_index": state.step_index, "retry_requested": True},
            )
            record_model_result(state, retry_result)
            reasoning = ReasoningStep.model_validate(retry_result.output.model_dump())
        return reasoning

    def _fallback_reasoning(self, state: AgentState) -> ReasoningStep:
        if state.decision is None:
            raise ValueError("Agent decision must exist before reasoning.")

        candidate_tools = [request.tool_name for request in state.decision.tool_requests]
        should_replan = not candidate_tools
        route_name = state.route.agent_name if state.route else "general"
        return ReasoningStep(
            objective=state.request.message,
            thought=(
                f"Use the {route_name} specialist lane to make measurable progress while staying aligned "
                "with the current plan, memory, and reflection constraints."
            ),
            reasoning_summary=(
                f"Step {state.step_index}: reason over {len(state.memory.retrieved)} retrieved memories, "
                f"{len(recent_observation_summaries(state))} recent observations, and "
                f"{len(candidate_tools)} candidate tool(s)."
            ),
            action_strategy=(
                "Prefer evidence-producing tools first, then summarize only what the observations support."
            ),
            candidate_tools=candidate_tools,
            selected_skills=list(state.decision.skill_names),
            expected_observation=(
                state.decision.expected_deliverables[0]
                if state.decision.expected_deliverables
                else "Fresh observable evidence for the current objective."
            ),
            should_replan=should_replan,
            replan_reason=(
                "The specialist did not produce any viable tool requests for the next action."
                if should_replan
                else None
            ),
            stop_signal="continue",
            stop_reason=None,
        )

    def _reasoning_prompt(self, state: AgentState, fallback: ReasoningStep) -> str:
        return "\n".join(
            [
                f"Objective: {state.request.message}",
                f"Architecture mode: {state.architecture.mode.value if state.architecture else ''}",
                f"Context packet: {state.context_packet.context_summary if state.context_packet else ''}",
                f"Plan summary: {state.plan.task_summary if state.plan else ''}",
                f"Current route: {state.route.agent_name if state.route else 'general'}",
                f"Decision summary: {state.decision.summary if state.decision else ''}",
                f"Candidate tools: {json.dumps([request.model_dump(mode='json') for request in (state.decision.tool_requests if state.decision else [])], ensure_ascii=True, default=str)}",
                f"Recent observations: {json.dumps(recent_observation_summaries(state, limit=4), ensure_ascii=True)}",
                f"Adaptive constraints: {json.dumps(state.adaptive_constraints, ensure_ascii=True)}",
                f"Blocked tools: {json.dumps(state.blocked_tools, ensure_ascii=True)}",
                f"Fallback reasoning: {fallback.model_dump_json()}",
            ]
        )
