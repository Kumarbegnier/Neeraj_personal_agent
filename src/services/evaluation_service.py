from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel

from src.providers import ProviderInvocationError, ProviderRegistry
from src.schemas.evaluation import EvaluationSummary, ProviderEvaluationResult
from src.schemas.routing import ModelProvider, ModelTaskType

from .model_router import ModelRouter
from .provider_request_factory import build_structured_provider_request
from .structured_outputs import estimate_response_completeness, parse_structured_response, structured_task_success


class EvaluationService:
    """Runs the same structured task across multiple providers and compares the results."""

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        model_router: ModelRouter,
    ) -> None:
        self._provider_registry = provider_registry
        self._model_router = model_router

    def compare_structured(
        self,
        *,
        task_type: ModelTaskType,
        system_prompt: str,
        user_prompt: str,
        output_type: type[BaseModel],
        providers: Iterable[ModelProvider] | None = None,
        metadata: dict[str, Any] | None = None,
        web_grounded: bool = False,
    ) -> EvaluationSummary:
        selected_providers = list(providers or ModelProvider)
        results: list[ProviderEvaluationResult] = []
        winner: ProviderEvaluationResult | None = None

        for provider in selected_providers:
            route = self._model_router.route(task_type, selected_provider=provider)
            request = build_structured_provider_request(
                task_type=task_type,
                model=route.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=output_type,
                metadata=metadata,
                web_grounded=web_grounded,
            )
            client = self._provider_registry.get(provider)
            notes: list[str] = []
            parsed_output: BaseModel | None = None
            structured_output_validity = False
            latency_ms = 0

            try:
                response = client.invoke(request)
                latency_ms = response.latency_ms
                parsed_output = parse_structured_response(response.content, output_type)
                structured_output_validity = True
                notes.append("Structured output validated against the declared schema.")
            except ProviderInvocationError as exc:
                notes.append(str(exc))
            except Exception as exc:  # pragma: no cover - defensive path
                notes.append(str(exc))

            completeness = estimate_response_completeness(parsed_output)
            success = structured_task_success(parsed_output, structured_output_validity)
            if not success and structured_output_validity:
                notes.append("The output validated structurally but did not look complete enough to count as a success.")

            result = ProviderEvaluationResult(
                provider=provider,
                model=route.model,
                structured_output_validity=structured_output_validity,
                latency_ms=latency_ms,
                task_success=success,
                response_completeness=completeness,
                notes=notes,
                used_fallback=False,
                metadata=dict(metadata or {}),
            )
            results.append(result)
            if winner is None or self._score(result) > self._score(winner):
                winner = result

        summary = self._summary(task_type, results, winner)
        return EvaluationSummary(
            task_type=task_type,
            results=results,
            winner=winner.provider if winner is not None else None,
            summary=summary,
        )

    def _score(self, result: ProviderEvaluationResult) -> float:
        latency_penalty = min(result.latency_ms / 10000, 0.2)
        return (
            (1.0 if result.structured_output_validity else 0.0)
            + (1.0 if result.task_success else 0.0)
            + result.response_completeness
            - latency_penalty
        )

    def _summary(
        self,
        task_type: ModelTaskType,
        results: list[ProviderEvaluationResult],
        winner: ProviderEvaluationResult | None,
    ) -> str:
        if not results:
            return f"No provider runs were recorded for task type '{task_type.value}'."
        if winner is None:
            return f"All provider runs for task type '{task_type.value}' failed."
        return (
            f"{winner.provider.value} performed best for task type '{task_type.value}' with "
            f"valid={winner.structured_output_validity}, success={winner.task_success}, "
            f"completeness={winner.response_completeness:.2f}, latency_ms={winner.latency_ms}."
        )
