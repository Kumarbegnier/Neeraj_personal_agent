from __future__ import annotations

from src.agents.catalog import get_agent_descriptor
from src.agents.base import BaseAgent


class CodingAgent(BaseAgent):
    def __init__(self) -> None:
        descriptor = get_agent_descriptor("coding")
        super().__init__(
            name=descriptor.display_name,
            description=descriptor.description,
            instructions="You are the coding specialist. Focus on modular design, verification, and implementation quality.",
            tool_names=descriptor.default_tools,
        )
