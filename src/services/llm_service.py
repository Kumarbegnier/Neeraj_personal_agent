from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel

from src.core.config import get_settings
from src.providers import ProviderInvocationError, ProviderRegistry
from src.schemas.routing import ModelProvider, ModelTaskType
from src.services.evaluation_service import EvaluationService
from src.services.model_router import ModelRouter
from src.services.modeling.types import (
    EvaluationTelemetry,
    GenerationTelemetry,
    StructuredGenerationResult,
)
from src.services.provider_request_factory import build_structured_provider_request
from src.services.routing_policy import RoutingPolicyService
from src.services.structured_outputs import (
    estimate_response_completeness,
    parse_structured_response,
    structured_task_success,
)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from agents import Agent as SDKAgent
    from agents import Runner
except Exception:  # pragma: no cover - optional dependency
    SDKAgent = None  # type: ignore[assignment]
    Runner = None  # type: ignore[assignment]


class LLMService:
    """Provider-agnostic orchestration runtime with OpenAI as the tool-calling backbone."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.chat_model = self.settings.openai_chat_model
        self.responses_model = self.settings.openai_responses_model
        self.embedding_model = self.settings.openai_embedding_model
        self.routing_policy = RoutingPolicyService(self.settings)
        self.model_router = ModelRouter(self.settings, self.routing_policy)
        self.provider_registry = ProviderRegistry(self.settings)
        self.evaluation_service = EvaluationService(
            provider_registry=self.provider_registry,
            model_router=self.model_router,
        )
        self._client = self._build_client()

    def _build_client(self) -> Any | None:
        if OpenAI is None or not self.settings.openai_api_key:
            return None
        return OpenAI(api_key=self.settings.openai_api_key)

    def info(self) -> dict[str, Any]:
        return {
            "providers": self._provider_health(),
            "routing_policy": self.model_router.routing_table(),
            "routing_entries": [entry.model_dump(mode="json") for entry in self.routing_policy.entries()],
            "tool_execution_provider": ModelProvider.OPENAI.value,
            "reflection_provider": ModelProvider.CLAUDE.value,
            "research_provider": ModelProvider.GEMINI.value,
            "planning_provider": ModelProvider.DEEPSEEK.value,
            "agents_sdk_available": self.agents_sdk_available(),
        }

    def health(self) -> dict[str, Any]:
        configured_providers = [name for name, details in self._provider_health().items() if details["configured"]]
        return {
            "configured": bool(configured_providers),
            "configured_providers": configured_providers,
            "agents_sdk_available": self.agents_sdk_available(),
            "providers": self._provider_health(),
            "routing_policy": self.model_router.routing_table(),
            "routing_entries": [entry.model_dump(mode="json") for entry in self.routing_policy.entries()],
        }

    def agents_sdk_available(self) -> bool:
        return bool(self.settings.use_openai_agents_sdk and SDKAgent is not None and Runner is not None)

    def generate_structured(
        self,
        *,
        task_type: ModelTaskType,
        stage: str,
        output_type: type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        fallback_output: BaseModel,
        selected_model: str | None = None,
        selected_provider: ModelProvider | None = None,
        metadata: dict[str, Any] | None = None,
        web_grounded: bool | None = None,
    ) -> StructuredGenerationResult[BaseModel]:
        route = self.model_router.route(
            task_type,
            selected_model=selected_model,
            selected_provider=selected_provider,
        )
        request = build_structured_provider_request(
            task_type=task_type,
            model=route.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=output_type,
            metadata=metadata,
            web_grounded=task_type == ModelTaskType.WEB_GROUNDING if web_grounded is None else web_grounded,
        )

        started = perf_counter()
        status = "fallback"
        source = "fallback"
        output = fallback_output
        reason = route.reason
        used_fallback = True
        latency_ms = 0
        notes: list[str] = []

        try:
            response = self.provider_registry.get(route.provider).invoke(request)
            latency_ms = response.latency_ms
            output = parse_structured_response(response.content, output_type)
            status = "success"
            source = "live"
            used_fallback = False
            reason = (
                f"{route.provider.value} returned schema-valid structured output for task "
                f"'{task_type.value}'."
            )
            notes.append("Structured output validated against the declared schema.")
        except ProviderInvocationError as exc:
            latency_ms = int((perf_counter() - started) * 1000)
            reason = str(exc)
            notes.append("Provider invocation failed and deterministic fallback was used.")
        except Exception as exc:  # pragma: no cover - defensive path
            latency_ms = int((perf_counter() - started) * 1000)
            status = "error"
            reason = str(exc)
            notes.append("Structured validation failed and deterministic fallback was used.")

        completeness = estimate_response_completeness(output)
        structured_output_validity = not used_fallback
        task_success = structured_task_success(output if structured_output_validity else None, structured_output_validity)
        score = self._evaluation_score(
            structured_output_validity=structured_output_validity,
            task_success=task_success,
            completeness=completeness,
            latency_ms=latency_ms,
        )

        run = GenerationTelemetry(
            task_type=task_type.value,
            stage=stage,
            provider=route.provider.value,
            model=route.model,
            status=status,
            source=source,
            latency_ms=latency_ms,
            used_fallback=used_fallback,
            reason=reason,
            candidate_models=list(route.candidate_models),
            metadata=metadata or {},
        )
        evaluation = EvaluationTelemetry(
            task_type=task_type.value,
            provider=route.provider.value,
            model=route.model,
            score=score,
            notes=notes,
            compared_models=list(route.candidate_models),
            metadata={
                "stage": stage,
                "used_fallback": used_fallback,
                **(metadata or {}),
            },
            structured_output_validity=structured_output_validity,
            latency_ms=latency_ms,
            task_success=task_success,
            response_completeness=completeness,
        )
        return StructuredGenerationResult(
            output=output,
            route=route,
            run=run,
            evaluation=evaluation,
        )

    def evaluate_structured(
        self,
        *,
        task_type: ModelTaskType,
        system_prompt: str,
        user_prompt: str,
        output_type: type[BaseModel],
        providers: list[ModelProvider] | None = None,
        metadata: dict[str, Any] | None = None,
        web_grounded: bool = False,
    ):
        return self.evaluation_service.compare_structured(
            task_type=task_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=output_type,
            providers=providers,
            metadata=metadata,
            web_grounded=web_grounded,
        )

    def build_sdk_agent(
        self,
        *,
        name: str,
        instructions: str,
        tools: list[Callable[..., Any]] | None = None,
        output_type: type[Any] | None = None,
        model: str | None = None,
    ) -> Any | None:
        """Build an OpenAI Agents SDK agent when the dependency is available."""
        if not self.agents_sdk_available():
            return None
        return SDKAgent(  # type: ignore[misc]
            name=name,
            instructions=instructions,
            tools=tools or [],
            output_type=output_type,
            model=model or self.responses_model,
        )

    def run_sdk_agent_sync(
        self,
        *,
        name: str,
        instructions: str,
        input_text: str,
        tools: list[Callable[..., Any]] | None = None,
        output_type: type[Any] | None = None,
    ) -> dict[str, Any]:
        """Run a small agent task when SDK + credentials are available, else return a structured noop."""
        agent = self.build_sdk_agent(
            name=name,
            instructions=instructions,
            tools=tools,
            output_type=output_type,
        )
        if agent is None or Runner is None:
            return {
                "status": "skipped",
                "reason": "OpenAI Agents SDK is unavailable or not configured.",
                "final_output": None,
            }
        try:  # pragma: no cover - networked path
            result = Runner.run_sync(agent, input_text)  # type: ignore[misc]
            return {
                "status": "success",
                "final_output": getattr(result, "final_output", None),
            }
        except Exception as exc:  # pragma: no cover - defensive path
            return {
                "status": "error",
                "reason": str(exc),
                "final_output": None,
            }

    def _provider_health(self) -> dict[str, dict[str, Any]]:
        return self.provider_registry.health()

    def _evaluation_score(
        self,
        *,
        structured_output_validity: bool,
        task_success: bool,
        completeness: float,
        latency_ms: int,
    ) -> float:
        latency_penalty = min(latency_ms / 10000, 0.2)
        score = (
            (1.0 if structured_output_validity else 0.0)
            + (1.0 if task_success else 0.0)
            + completeness
            - latency_penalty
        )
        return round(max(score, 0.0), 2)
