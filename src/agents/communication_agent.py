from __future__ import annotations

from src.agents.catalog import get_agent_descriptor
from src.agents.base import BaseAgent


class CommunicationAgent(BaseAgent):
    def __init__(self) -> None:
        descriptor = get_agent_descriptor("communication")
        super().__init__(
            name=descriptor.display_name,
            description=descriptor.description,
            instructions="You are the communication specialist. Produce structured, approval-aware communication drafts.",
            tool_names=descriptor.default_tools,
        )
