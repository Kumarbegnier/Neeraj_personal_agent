from __future__ import annotations

from typing import Any, Callable

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
    """Thin OpenAI / Agents SDK integration layer with graceful local fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.chat_model = self.settings.openai_chat_model
        self.responses_model = self.settings.openai_responses_model
        self.embedding_model = self.settings.openai_embedding_model
        self._client = self._build_client()

    def _build_client(self) -> Any | None:
        if OpenAI is None or not self.settings.openai_api_key:
            return None
        return OpenAI(api_key=self.settings.openai_api_key)

    def info(self) -> dict[str, Any]:
        return {
            "chat_model": self.chat_model,
            "responses_model": self.responses_model,
            "embedding_model": self.embedding_model,
            "openai_configured": self._client is not None,
            "agents_sdk_available": self.agents_sdk_available(),
            "provider": "openai_responses_compatible",
        }

    def health(self) -> dict[str, Any]:
        return {
            "configured": self._client is not None,
            "agents_sdk_available": self.agents_sdk_available(),
            "responses_model": self.responses_model,
        }

    def agents_sdk_available(self) -> bool:
        return bool(self.settings.use_openai_agents_sdk and SDKAgent is not None and Runner is not None)

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
