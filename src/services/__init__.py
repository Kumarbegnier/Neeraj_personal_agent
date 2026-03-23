from .llm_service import LLMService

__all__ = ["LLMService", "OrchestrationService"]


def __getattr__(name: str):
    if name == "OrchestrationService":
        from .orchestration_service import OrchestrationService

        return OrchestrationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
