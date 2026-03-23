from __future__ import annotations

import json

from src.services.llm_service import LLMService
from src.services.modeling.types import ModelTaskType

from .models import AgentState, ControlDecision
from .runtime_utils import (
    lowercase_surface,
    recent_observation_summaries,
    record_model_result,
    selected_model_override,
)


class OrchestratorBrain:
    def __init__(self, llm_service: LLMService | None = None) -> None:
        self._llm_service = llm_service

    def decide(self, state: AgentState) -> ControlDecision:
        fallback = self._fallback_decision(state)
        if self._llm_service is None:
            return fallback

        context = state.context
        model_result = self._llm_service.generate_structured(
            task_type=ModelTaskType.ORCHESTRATION,
            stage="control",
            output_type=ControlDecision,
            system_prompt=(
                "You are the orchestration control layer for a hybrid multi-model agent runtime. "
                "Produce a concise control decision that is safe, stateful, and tool-aware."
            ),
            user_prompt="\n".join(
                [
                    f"User message: {state.request.message}",
                    f"Active goals: {json.dumps(context.active_goals if context else [], ensure_ascii=True)}",
                    f"Requested capabilities: {json.dumps(context.requested_capabilities if context else [], ensure_ascii=True)}",
                    f"Recent observations: {json.dumps(recent_observation_summaries(state, limit=4), ensure_ascii=True)}",
                    f"Adaptive constraints: {json.dumps(state.adaptive_constraints, ensure_ascii=True)}",
                    f"Fallback decision: {fallback.model_dump_json()}",
                ]
            ),
            fallback_output=fallback,
            selected_model=selected_model_override(state),
            metadata={"step_index": state.step_index},
        )
        record_model_result(state, model_result)
        return ControlDecision.model_validate(model_result.output.model_dump())

    def _fallback_decision(self, state: AgentState) -> ControlDecision:
        context = state.context
        if context is None:
            raise ValueError("Context must be available before control decisions.")

        lowered = self._state_surface(state)
        memory_bias = self._memory_bias(state)
        intent = self._derive_intent(lowered, context.active_goals, context.requested_capabilities, memory_bias)
        urgency = "high" if any(word in lowered for word in ("urgent", "asap", "immediately")) else "normal"
        preferred_agent = self._preferred_agent(lowered, context.active_goals, context.requested_capabilities, state, memory_bias)

        complexity = context.signals.complexity
        risk_level = context.signals.risk_level
        reasoning_mode = "deliberate-multistep" if complexity == "high" or state.needs_replan else "directed"
        coordination_pattern = (
            "modular-orchestration"
            if context.signals.collaboration_mode == "modular-orchestration" or state.route_bias is not None
            else "single-specialist"
        )
        verification_mode = (
            "strict"
            if complexity == "high" or "verification" in context.requested_capabilities or state.retry_count > 0
            else "standard"
        )
        needs_tooling = any(
            capability in context.requested_capabilities
            for capability in ("coding", "research", "browser", "documents", "verification", "memory")
        ) or state.retry_count > 0

        notes = (
            f"Step {state.step_index}: interpreted intent '{intent}' with complexity '{complexity}', "
            f"risk '{risk_level}', and preferred agent '{preferred_agent or 'general'}'. "
            "Control factored in retrieved memory, recent observations, prior verification, and reflection constraints."
        )

        return ControlDecision(
            intent=intent,
            control_notes=notes,
            preferred_agent=preferred_agent,
            urgency=urgency,
            complexity=complexity,
            reasoning_mode=reasoning_mode,
            llm_role="planner-and-synthesizer",
            coordination_pattern=coordination_pattern,
            risk_level=risk_level,
            memory_strategy="retrieve-rank-summarize",
            verification_mode=verification_mode,
            needs_tooling=needs_tooling,
        )

    def _derive_intent(
        self,
        lowered: str,
        goals: list[str],
        capabilities: list[str],
        memory_bias: dict[str, float],
    ) -> str:
        if "design_system" in goals or "coding" in capabilities or memory_bias.get("coding", 0.0) > 1.5:
            return "system_design"
        if "research" in goals or memory_bias.get("research", 0.0) > 1.5:
            return "investigation"
        if "communicate" in goals or memory_bias.get("communication", 0.0) > 1.5:
            return "communication"
        if "manage_tasks" in goals:
            return "task_management"
        if "inspect_files" in goals:
            return "document_analysis"
        if "verification" in capabilities or "verify_outputs" in goals:
            return "verification"
        if "memory" in capabilities or "maintain_memory" in goals:
            return "memory_grounding"
        if "architecture" in lowered or "orchestrator" in lowered:
            return "system_design"
        return "general_assistance"

    def _preferred_agent(
        self,
        lowered: str,
        goals: list[str],
        capabilities: list[str],
        state: AgentState,
        memory_bias: dict[str, float],
    ) -> str | None:
        if state.route_bias:
            return state.route_bias

        scored = {
            "communication": memory_bias.get("communication", 0.0),
            "task": memory_bias.get("task", 0.0),
            "file": memory_bias.get("file", 0.0),
            "web": memory_bias.get("web", 0.0),
            "research": memory_bias.get("research", 0.0),
            "coding": memory_bias.get("coding", 0.0),
            "general": 0.1,
        }

        if "communicate" in goals or "communication" in capabilities:
            scored["communication"] += 2.5
        if "manage_tasks" in goals or "coordination" in capabilities:
            scored["task"] += 2.2
        if "inspect_files" in goals or "documents" in capabilities:
            scored["file"] += 2.2
        if "browse_web" in goals or "browser" in capabilities:
            scored["web"] += 2.0
        if "research" in goals and "coding" not in capabilities:
            scored["research"] += 2.4
        if "design_system" in goals or "ship_code" in goals:
            scored["coding"] += 2.6
        if any(word in lowered for word in ("architecture", "orchestrator", "system", "build", "code")):
            scored["coding"] += 1.8

        if state.retry_count > 0:
            scored["coding"] += 0.5
            scored["general"] -= 0.2

        winner = max(scored.items(), key=lambda item: item[1])[0]
        return winner if winner != "general" or max(scored.values()) > 0.5 else "general"

    def _state_surface(self, state: AgentState) -> str:
        observation_text = " ".join(recent_observation_summaries(state))
        reflection_text = " ".join(state.reflection.repairs if state.reflection else [])
        constraint_text = " ".join(state.adaptive_constraints)
        focus_text = state.execution.next_focus if state.execution else ""
        memory_text = " ".join(record.content for record in state.memory.retrieved[:4])
        return lowercase_surface(
            [
                state.request.message,
                observation_text,
                reflection_text,
                constraint_text,
                focus_text,
                memory_text,
            ]
        )

    def _memory_bias(self, state: AgentState) -> dict[str, float]:
        bias = {
            "communication": 0.0,
            "task": 0.0,
            "file": 0.0,
            "web": 0.0,
            "research": 0.0,
            "coding": 0.0,
        }
        for record in state.memory.retrieved:
            text = record.content.lower()
            tags = {tag.lower() for tag in record.tags}
            if "last assigned agent" in text:
                for agent in bias:
                    if agent in text:
                        bias[agent] += 1.2
            if "verification" in text or "verification" in tags:
                bias["coding"] += 0.4
                bias["research"] += 0.2
            if "reflection lesson" in text:
                bias["coding"] += 0.3
            if any(word in text for word in ("document", "pdf", "report")):
                bias["file"] += 0.8
            if any(word in text for word in ("research", "evidence", "summary")):
                bias["research"] += 0.8
            if any(word in text for word in ("email", "message", "reply")):
                bias["communication"] += 0.8
            if any(word in text for word in ("calendar", "task", "meeting")):
                bias["task"] += 0.8
        return bias
