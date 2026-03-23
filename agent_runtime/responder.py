from __future__ import annotations

from .models import AgentState


class ResponseComposer:
    def compose(self, state: AgentState) -> str:
        return self.synthesize_from_state(state)

    def synthesize_from_state(self, state: AgentState) -> str:
        agent_name = state.route.agent_name if state.route else "general"
        execution_summary = state.execution.summary if state.execution else "No execution summary was produced."
        observation_clause = self._observation_clause(state)
        verification_clause = self._verification_clause(state)
        reflection_clause = self._reflection_clause(state)
        safety_clause = self._safety_clause(state)

        return " ".join(
            part
            for part in [
                f"The {agent_name} agent completed the request after {state.step_index} loop step(s).",
                execution_summary,
                observation_clause,
                verification_clause,
                reflection_clause,
                safety_clause,
            ]
            if part
        ).strip()

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

    def _safety_clause(self, state: AgentState) -> str:
        if not state.safety:
            return ""
        if state.safety.permission.requires_confirmation:
            return "Additional confirmation is required before any side effect."
        return "No additional approval gate was triggered."
