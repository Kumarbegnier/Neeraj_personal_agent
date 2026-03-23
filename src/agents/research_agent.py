from __future__ import annotations

from src.agents.catalog import get_agent_descriptor
from src.agents.base import BaseAgent


class ResearchAgent(BaseAgent):
    def __init__(self) -> None:
        descriptor = get_agent_descriptor("research")
        super().__init__(
            name=descriptor.display_name,
            description=descriptor.description,
            instructions="You are the research specialist. Collect evidence, ground claims, and summarize findings clearly.",
            tool_names=descriptor.default_tools,
        )
