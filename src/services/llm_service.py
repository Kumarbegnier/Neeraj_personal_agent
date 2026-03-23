from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel

from src.services.modeling.providers import (
    ClaudeProviderAdapter,
    DeepSeekProviderAdapter,
    GeminiProviderAdapter,
    OpenAIProviderAdapter,
    ProviderInvocationError,
)
from src.services.modeling.router import ModelRouter
from src.services.modeling.types import (
    EvaluationTelemetry,
    GenerationTelemetry,
    ModelProvider,
    ModelTaskType,
    StructuredGenerationResult,
)
from src.core.config import get_settings

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
    """Hybrid multi-provider model runtime with structured-output routing and OpenAI SDK support."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.chat_model = self.settings.openai_chat_model
        self.responses_model = self.settings.openai_responses_model
        self.embedding_model = self.settings.openai_embedding_model
        self.model_router = ModelRouter(self.settings)
        self._providers = {
            ModelProvider.OPENAI: OpenAIProviderAdapter(self.settings),
            ModelProvider.CLAUDE: ClaudeProviderAdapter(self.settings),
            ModelProvider.GEMINI: GeminiProviderAdapter(self.settings),
            ModelProvider.DEEPSEEK: DeepSeekProviderAdapter(self.settings),
        }
        self._client = self._build_client()

    def _build_client(self) -> Any | None:
        if OpenAI is None or not self.settings.openai_api_key:
            return None
        return OpenAI(api_key=self.settings.openai_api_key)

    def info(self) -> dict[str, Any]:
        return {
            "providers": self._provider_health(),
            "routing_policy": self.model_router.routing_table(),
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
        metadata: dict[str, Any] | None = None,
    ) -> StructuredGenerationResult[BaseModel]:
        route = self.model_router.route(task_type, selected_model=selected_model)
        provider = self._providers[route.provider]
        started = perf_counter()
        status = "fallback"
        source = "fallback"
        reason = route.reason
        used_fallback = True
        output = fallback_output

        try:
            live_output = provider.generate_structured(
                model=route.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=output_type,
            )
            output = live_output
            status = "success"
            source = "live"
            reason = f"{route.provider.value} returned structured output for task '{task_type.value}'."
            used_fallback = False
        except ProviderInvocationError as exc:
            reason = str(exc)
        except Exception as exc:  # pragma: no cover - defensive path
            status = "error"
            reason = str(exc)

        latency_ms = int((perf_counter() - started) * 1000)
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
            candidate_models=route.candidate_models,
            metadata=metadata or {},
        )
        evaluation_notes = (
            "Structured output validated against the declared schema."
            if not used_fallback
            else "Deterministic fallback preserved hybrid execution semantics."
        )
        evaluation = EvaluationTelemetry(
            task_type=task_type.value,
            provider=route.provider.value,
            model=route.model,
            score=1.0 if not used_fallback else 0.45,
            notes=(evaluation_notes,),
            compared_models=route.candidate_models,
            metadata={
                "stage": stage,
                "used_fallback": used_fallback,
                **(metadata or {}),
            },
        )
        return StructuredGenerationResult(
            output=output,
            route=route,
            run=run,
            evaluation=evaluation,
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
        return {
            provider.value: adapter.health()
            for provider, adapter in self._providers.items()
        }
