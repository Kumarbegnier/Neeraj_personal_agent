from .base import BaseAgent
from .coding import CodingAgent
from .communication import CommunicationAgent
from .file import FileAgent
from .general import GeneralAgent
from .research import ResearchAgent
from .task import TaskAgent
from .web import WebAgent

from src.services.llm_service import LLMService


def build_agent_registry(llm_service: LLMService | None = None) -> dict[str, BaseAgent]:
    agents: list[BaseAgent] = [
        CommunicationAgent(llm_service),
        CodingAgent(llm_service),
        ResearchAgent(llm_service),
        WebAgent(llm_service),
        TaskAgent(llm_service),
        FileAgent(llm_service),
        GeneralAgent(llm_service),
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
