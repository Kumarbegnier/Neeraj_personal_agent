from __future__ import annotations

from src.core.config import Settings

from .types import ModelProvider, ModelRoute, ModelTaskType


class ModelRouter:
    """Routes model tasks to providers without leaking provider logic into runtime stages."""

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

    def route(
        self,
        task_type: ModelTaskType,
        *,
        selected_model: str | None = None,
    ) -> ModelRoute:
        default_provider = self._routing_policy.get(task_type, ModelProvider.OPENAI)
        default_model = self.default_model(default_provider)
        explicit_override = self._is_explicit_override(selected_model)
        override_provider = self._provider_for_model(selected_model) if explicit_override else None
        normalized_model = self._normalize_model_name(selected_model) if selected_model else None
        provider = override_provider or default_provider
        model = (
            normalized_model
            if normalized_model and (override_provider is not None or normalized_model == default_model)
            else self.default_model(provider)
        )
        if override_provider and selected_model:
            reason = f"Explicit provider override '{selected_model}' mapped to provider '{provider.value}'."
        elif normalized_model and normalized_model == default_model:
            reason = (
                f"Task type '{task_type.value}' kept its default provider '{provider.value}' and honored "
                f"the matching model selection '{normalized_model}'."
            )
        else:
            reason = f"Task type '{task_type.value}' routes to '{provider.value}' by the hybrid execution policy."
        return ModelRoute(
            task_type=task_type,
            provider=provider,
            model=model,
            reason=reason,
            candidate_models=self.candidate_models(),
        )

    def default_model(self, provider: ModelProvider) -> str:
        return {
            ModelProvider.OPENAI: self._settings.openai_responses_model,
            ModelProvider.CLAUDE: self._settings.claude_model,
            ModelProvider.GEMINI: self._settings.gemini_model,
            ModelProvider.DEEPSEEK: self._settings.deepseek_model,
        }[provider]

    def candidate_models(self) -> tuple[str, ...]:
        return (
            self._settings.openai_responses_model,
            self._settings.claude_model,
            self._settings.gemini_model,
            self._settings.deepseek_model,
        )

    def routing_table(self) -> dict[str, str]:
        return {task.value: provider.value for task, provider in self._routing_policy.items()}

    def _provider_for_model(self, model_name: str | None) -> ModelProvider | None:
        if not model_name:
            return None

        lowered = model_name.strip().lower()
        if ":" in lowered:
            prefix = lowered.split(":", 1)[0]
            if prefix in {provider.value for provider in ModelProvider}:
                return ModelProvider(prefix)
        if "/" in lowered:
            prefix = lowered.split("/", 1)[0]
            if prefix in {provider.value for provider in ModelProvider}:
                return ModelProvider(prefix)
        if lowered.startswith(("gpt-", "o1", "o3", "openai")):
            return ModelProvider.OPENAI
        if lowered.startswith(("claude", "anthropic")):
            return ModelProvider.CLAUDE
        if lowered.startswith("gemini"):
            return ModelProvider.GEMINI
        if lowered.startswith("deepseek"):
            return ModelProvider.DEEPSEEK
        return None

    def _is_explicit_override(self, model_name: str | None) -> bool:
        if not model_name:
            return False
        lowered = model_name.strip().lower()
        return ":" in lowered or "/" in lowered

    def _normalize_model_name(self, model_name: str | None) -> str | None:
        if not model_name:
            return None
        raw = model_name.strip()
        if ":" in raw:
            prefix, remainder = raw.split(":", 1)
            if prefix.lower() in {provider.value for provider in ModelProvider} and remainder:
                return remainder.strip()
        if "/" in raw:
            prefix, remainder = raw.split("/", 1)
            if prefix.lower() in {provider.value for provider in ModelProvider} and remainder:
                return remainder.strip()
        return raw
