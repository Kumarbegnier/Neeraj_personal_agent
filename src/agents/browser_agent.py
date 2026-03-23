from __future__ import annotations

from src.agents.catalog import get_agent_descriptor
from src.agents.base import BaseAgent


class BrowserAgent(BaseAgent):
    def __init__(self) -> None:
        descriptor = get_agent_descriptor("web")
        super().__init__(
            name=descriptor.display_name,
            description=descriptor.description,
            instructions="You are the browser specialist. Use structured browser observations and stay approval-aware.",
            tool_names=descriptor.default_tools,
        )
