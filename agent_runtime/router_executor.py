from __future__ import annotations

from .agents import BaseAgent
from .models import AgentDecision, AgentRoute, AgentState, SkillDescriptor
from .router import AgentRouter
from .skills import SkillLibrary


class RouterExecutor:
    def __init__(
        self,
        agent_router: AgentRouter,
        skill_library: SkillLibrary,
        agents: dict[str, BaseAgent],
    ) -> None:
        self._agent_router = agent_router
        self._skill_library = skill_library
        self._agents = agents

    def route(self, state: AgentState) -> AgentRoute:
        return self._agent_router.route(state)

    def skills_for(
        self,
        state: AgentState,
        agent_name: str,
    ) -> list[SkillDescriptor]:
        return self._skill_library.recommend(state.request, state.context, agent_name)

    def decide(
        self,
        state: AgentState,
        agent_name: str,
        skills: list[SkillDescriptor] | None = None,
    ) -> tuple[AgentDecision, BaseAgent]:
        resolved_skills = skills if skills is not None else self.skills_for(state, agent_name)
        agent = self.agent_for(agent_name)
        decision = agent.decide(state, resolved_skills)
        return decision, agent

    def prepare(
        self,
        state: AgentState,
    ) -> tuple[AgentRoute, list[SkillDescriptor], AgentDecision, BaseAgent]:
        route = self.route(state)
        skills = self.skills_for(state, route.agent_name)
        decision, agent = self.decide(state, route.agent_name, skills)
        return route, skills, decision, agent

    def agent_for(self, agent_name: str) -> BaseAgent:
        return self._agents.get(agent_name, self._agents["general"])
