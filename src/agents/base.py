from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.agents.catalog import get_agent_descriptor
from src.services.llm_service import LLMService


class AgentInput(BaseModel):
    objective: str
    context: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    summary: str
    deliverables: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class BaseAgent(BaseModel):
    name: str
    description: str
    instructions: str
    tool_names: list[str] = Field(default_factory=list)

    @classmethod
    def from_catalog(cls, key: str, *, instructions: str) -> "BaseAgent":
        descriptor = get_agent_descriptor(key)
        return cls(
            name=descriptor.display_name,
            description=descriptor.description,
            instructions=instructions,
            tool_names=list(descriptor.default_tools),
        )

    def build_sdk_agent(self, llm_service: LLMService):
        return llm_service.build_sdk_agent(
            name=self.name,
            instructions=self.instructions,
            tools=[],
            output_type=AgentOutput,
        )

    def run(self, llm_service: LLMService, objective: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return llm_service.run_sdk_agent_sync(
            name=self.name,
            instructions=self.instructions,
            input_text=AgentInput(objective=objective, context=context or {}).model_dump_json(),
            tools=[],
            output_type=AgentOutput,
        )
