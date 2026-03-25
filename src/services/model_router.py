from __future__ import annotations

from src.core.config import Settings
from src.schemas.evaluation import ProviderEvaluationResult
from src.schemas.routing import (
    HistoricalRoutingStats,
    ModelProvider,
    ModelTaskType,
    RoutingDecision,
    RoutingRequest,
    TaskFamilyRoutingWinner,
)

from .routing_policy import RoutingPolicyService


class ModelRouter:
    """Routes model tasks through a clean policy service without leaking provider details downstream."""

    def __init__(
        self,
        settings: Settings,
        routing_policy: RoutingPolicyService | None = None,
    ) -> None:
        self._settings = settings
        self._routing_policy = routing_policy or RoutingPolicyService(settings)

    def route(
        self,
        task_or_request: ModelTaskType | RoutingRequest,
        *,
        selected_model: str | None = None,
        selected_provider: ModelProvider | None = None,
    ) -> RoutingDecision:
        request = (
            task_or_request
            if isinstance(task_or_request, RoutingRequest)
            else RoutingRequest(
                task_type=task_or_request,
                selected_model=selected_model,
                selected_provider=selected_provider,
            )
        )
        default_entry = self._routing_policy.entry(request.task_type)
        adaptive_decision = self._routing_policy.decide(request)
        explicit_provider = self._provider_for_model(request.selected_model)
        provider = explicit_provider or request.selected_provider or adaptive_decision.selected_provider
        normalized_model = self._normalize_model_name(request.selected_model)
        model = normalized_model or self._routing_policy.default_model(provider)

        if explicit_provider and request.selected_model:
            reason = (
                f"Explicit model override '{request.selected_model}' mapped to provider "
                f"'{provider.value}'."
            )
        elif request.selected_provider is not None and request.selected_provider != default_entry.provider:
            reason = (
                f"Routing request overrode the default provider '{default_entry.provider.value}' "
                f"with '{request.selected_provider.value}'."
            )
        elif normalized_model and model != default_entry.default_model:
            reason = (
                f"Task type '{request.task_type.value}' stayed on '{provider.value}' while honoring "
                f"the explicit model selection '{model}'."
            )
        else:
            reason = adaptive_decision.reason

        return RoutingDecision(
            task_type=request.task_type,
            task_family=adaptive_decision.task_family,
            provider=provider,
            model=model,
            reason=reason,
            fallback_provider=adaptive_decision.fallback_provider,
            adaptive_decision=adaptive_decision,
            candidate_models=self._routing_policy.candidate_models(),
        )

    def default_model(self, provider: ModelProvider) -> str:
        return self._routing_policy.default_model(provider)

    def candidate_models(self) -> list[str]:
        return self._routing_policy.candidate_models()

    def routing_table(self) -> dict[str, str]:
        return self._routing_policy.routing_table()

    def record_evaluation(self, result: ProviderEvaluationResult) -> None:
        self._routing_policy.record_evaluation(result)

    def historical_stats(
        self,
        *,
        task_type: ModelTaskType,
        task_family: str,
    ) -> list[HistoricalRoutingStats]:
        return self._routing_policy.historical_stats(task_type=task_type, task_family=task_family)

    def adaptive_routing_health(self) -> dict[str, object]:
        return self._routing_policy.health()

    def evaluation_winners(self, *, limit: int = 12) -> list[TaskFamilyRoutingWinner]:
        return self._routing_policy.evaluation_winners(limit=limit)

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
        if lowered.startswith(("gpt-", "o1", "o3", "o4", "openai")):
            return ModelProvider.OPENAI
        if lowered.startswith(("claude", "anthropic")):
            return ModelProvider.CLAUDE
        if lowered.startswith("gemini"):
            return ModelProvider.GEMINI
        if lowered.startswith("deepseek"):
            return ModelProvider.DEEPSEEK
        return None

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
