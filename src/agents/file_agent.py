from __future__ import annotations

from src.agents.catalog import get_agent_descriptor
from src.agents.base import BaseAgent


class FileAgent(BaseAgent):
    def __init__(self) -> None:
        descriptor = get_agent_descriptor("file")
        super().__init__(
            name=descriptor.display_name,
            description=descriptor.description,
            instructions="You are the file specialist. Extract, summarize, and ground document analysis in evidence.",
            tool_names=descriptor.default_tools,
        )
