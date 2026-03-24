from __future__ import annotations

from src.core.config import Settings
from src.schemas.routing import ModelProvider, ModelTaskType, RoutingPolicyEntry


class RoutingPolicyService:
    """Owns the default provider-selection policy for the multi-model runtime."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._routing_policy = {
            ModelTaskType.ORCHESTRATION: ModelProvider.OPENAI,
            ModelTaskType.TOOL_EXECUTION: ModelProvider.OPENAI,
            ModelTaskType.COMMUNICATION: ModelProvider.CLAUDE,
            ModelTaskType.REFLECTION: ModelProvider.CLAUDE,
            ModelTaskType.RESEARCH: ModelProvider.GEMINI,
            ModelTaskType.WEB_GROUNDING: ModelProvider.GEMINI,
            ModelTaskType.PLANNING: ModelProvider.DEEPSEEK,
            ModelTaskType.REASONING: ModelProvider.DEEPSEEK,
        }
        self._rationales = {
            ModelTaskType.ORCHESTRATION: "OpenAI remains the orchestration and tool-calling backbone.",
            ModelTaskType.TOOL_EXECUTION: "OpenAI is the default execution summarizer and tool-calling path.",
            ModelTaskType.COMMUNICATION: "Claude is preferred for polished communication-heavy outputs.",
            ModelTaskType.REFLECTION: "Claude is preferred for critique, reflection, and revision guidance.",
            ModelTaskType.RESEARCH: "Gemini is preferred for research-oriented synthesis tasks.",
            ModelTaskType.WEB_GROUNDING: "Gemini is preferred when web grounding is part of the task.",
            ModelTaskType.PLANNING: "DeepSeek is preferred for structured planning and decomposition.",
            ModelTaskType.REASONING: "DeepSeek is preferred for JSON-heavy reasoning and verification.",
        }

    def provider_for(self, task_type: ModelTaskType) -> ModelProvider:
        return self._routing_policy.get(task_type, ModelProvider.OPENAI)

    def default_model(self, provider: ModelProvider) -> str:
        return {
            ModelProvider.OPENAI: self._settings.openai_responses_model,
            ModelProvider.CLAUDE: self._settings.claude_model,
            ModelProvider.GEMINI: self._settings.gemini_model,
            ModelProvider.DEEPSEEK: self._settings.deepseek_model,
        }[provider]

    def candidate_models(self) -> list[str]:
        return [
            self._settings.openai_responses_model,
            self._settings.claude_model,
            self._settings.gemini_model,
            self._settings.deepseek_model,
        ]

    def entry(self, task_type: ModelTaskType) -> RoutingPolicyEntry:
        provider = self.provider_for(task_type)
        return RoutingPolicyEntry(
            task_type=task_type,
            provider=provider,
            default_model=self.default_model(provider),
            rationale=self._rationales.get(task_type, "Defaulted to the primary orchestration provider."),
        )

    def entries(self) -> list[RoutingPolicyEntry]:
        return [self.entry(task_type) for task_type in ModelTaskType]

    def routing_table(self) -> dict[str, str]:
        return {task_type.value: self.provider_for(task_type).value for task_type in ModelTaskType}
