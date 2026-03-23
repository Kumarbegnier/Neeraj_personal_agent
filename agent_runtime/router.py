from __future__ import annotations

from .models import AgentRoute, AgentState
from .runtime_utils import lowercase_surface, recent_observation_summaries


class AgentRouter:
    def route(self, state: AgentState) -> AgentRoute:
        context = state.context
        control = state.control
        plan = state.plan
        if context is None or control is None or plan is None:
            raise ValueError("Context, control, and plan must exist before routing.")

        if state.route_bias:
            return AgentRoute(
                agent_name=state.route_bias,
                rationale=f"Reflection biased the next route toward '{state.route_bias}'.",
                confidence=0.78,
            )

        lowered = self._routing_surface(state)
        scores = {
            "communication": 0.0,
            "coding": 0.0,
            "research": 0.0,
            "web": 0.0,
            "task": 0.0,
            "file": 0.0,
            "general": 0.2,
        }

        if control.preferred_agent:
            scores[control.preferred_agent] = scores.get(control.preferred_agent, 0.0) + 2.0

        for capability in context.requested_capabilities:
            if capability == "communication":
                scores["communication"] += 1.8
            elif capability == "coding":
                scores["coding"] += 1.8
            elif capability == "research":
                scores["research"] += 1.8
            elif capability == "browser":
                scores["web"] += 1.6
            elif capability == "coordination":
                scores["task"] += 1.6
            elif capability == "documents":
                scores["file"] += 1.6

        memory_scores = self._memory_scores(state)
        for agent_name, score in memory_scores.items():
            scores[agent_name] = scores.get(agent_name, 0.0) + score

        if any(word in lowered for word in ("email", "message", "reply", "draft", "send")):
            scores["communication"] += 1.4
        if any(word in lowered for word in ("code", "coding", "debug", "bug", "fix", "implement", "api", "backend", "frontend", "architecture", "system")):
            scores["coding"] += 1.4
        if any(word in lowered for word in ("research", "search", "lookup", "compare", "summary")):
            scores["research"] += 1.4
        if any(word in lowered for word in ("browser", "web", "website", "page")):
            scores["web"] += 1.2
        if any(word in lowered for word in ("schedule", "calendar", "meeting", "remind", "task")):
            scores["task"] += 1.2
        if any(word in lowered for word in ("file", "pdf", "document", "report", "doc")):
            scores["file"] += 1.2

        if state.retry_count > 0:
            scores["general"] -= 0.3
            if state.reflection and "unsupported" in " ".join(state.reflection.repairs).lower():
                scores["research"] += 0.4
                scores["coding"] += 0.4

        winner, confidence = max(scores.items(), key=lambda item: item[1])
        rationale = (
            f"Selected '{winner}' using current capabilities {context.requested_capabilities}, "
            f"memory scores {memory_scores}, and plan strategy '{plan.decomposition_strategy}'."
        )
        return AgentRoute(agent_name=winner, rationale=rationale, confidence=round(min(0.95, 0.4 + confidence / 5), 2))

    def _routing_surface(self, state: AgentState) -> str:
        observation_text = " ".join(recent_observation_summaries(state))
        planning_text = " ".join(state.plan.constraints if state.plan else [])
        focus_text = state.execution.next_focus if state.execution else ""
        memory_text = " ".join(record.content for record in state.memory.retrieved[:4])
        return lowercase_surface(
            [
                state.request.message,
                observation_text,
                planning_text,
                focus_text,
                memory_text,
            ]
        )

    def _memory_scores(self, state: AgentState) -> dict[str, float]:
        scores = {
            "communication": 0.0,
            "coding": 0.0,
            "research": 0.0,
            "web": 0.0,
            "task": 0.0,
            "file": 0.0,
        }
        for record in state.memory.retrieved:
            text = record.content.lower()
            tags = {tag.lower() for tag in record.tags}
            if "last assigned agent" in text:
                for agent_name in scores:
                    if agent_name in text:
                        scores[agent_name] += 1.0
            if "reflection lesson" in text:
                scores["coding"] += 0.3
            if "verification summary" in text:
                scores["research"] += 0.2
                scores["coding"] += 0.2
            if any(word in text or word in tags for word in ("document", "pdf", "report")):
                scores["file"] += 0.7
            if any(word in text or word in tags for word in ("research", "evidence", "compare")):
                scores["research"] += 0.7
            if any(word in text or word in tags for word in ("email", "message", "reply")):
                scores["communication"] += 0.7
            if any(word in text or word in tags for word in ("browser", "website", "web")):
                scores["web"] += 0.6
            if any(word in text or word in tags for word in ("task", "calendar", "meeting")):
                scores["task"] += 0.6
        return scores
