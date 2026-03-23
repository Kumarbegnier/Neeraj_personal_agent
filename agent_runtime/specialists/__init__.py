from .base import BaseAgent
from .coding import CodingAgent
from .communication import CommunicationAgent
from .file import FileAgent
from .general import GeneralAgent
from .research import ResearchAgent
from .task import TaskAgent
from .web import WebAgent


def build_agent_registry() -> dict[str, BaseAgent]:
    agents: list[BaseAgent] = [
        CommunicationAgent(),
        CodingAgent(),
        ResearchAgent(),
        WebAgent(),
        TaskAgent(),
        FileAgent(),
        GeneralAgent(),
    ]
    return {agent.name: agent for agent in agents}


__all__ = [
    "BaseAgent",
    "CodingAgent",
    "CommunicationAgent",
    "FileAgent",
    "GeneralAgent",
    "ResearchAgent",
    "TaskAgent",
    "WebAgent",
    "build_agent_registry",
]
