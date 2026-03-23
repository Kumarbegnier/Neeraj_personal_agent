from __future__ import annotations

from src.agents.catalog import get_agent_descriptor
from src.agents.base import BaseAgent


class TaskAgent(BaseAgent):
    def __init__(self) -> None:
        descriptor = get_agent_descriptor("task")
        super().__init__(
            name=descriptor.display_name,
            description=descriptor.description,
            instructions="You are the task specialist. Track tasks clearly, preserve context, and keep actions reviewable.",
            tool_names=descriptor.default_tools,
        )
