from __future__ import annotations

from src.agents.catalog import get_agent_descriptor
from src.agents.base import BaseAgent


class GeneralAgent(BaseAgent):
    def __init__(self) -> None:
        descriptor = get_agent_descriptor("general")
        super().__init__(
            name=descriptor.display_name,
            description=descriptor.description,
            instructions="You are the general specialist. Handle mixed requests and keep the system moving when no narrower lane dominates.",
            tool_names=descriptor.default_tools,
        )
