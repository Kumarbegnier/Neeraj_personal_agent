from __future__ import annotations

import json
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

import httpx

from src.core.config import Settings
from src.schemas.provider import ProviderHealth, ProviderMessage, ProviderRequest, ProviderResponse
from src.schemas.routing import ModelProvider


class ProviderInvocationError(RuntimeError):
    pass


class BaseProviderClient(ABC):
    provider: ModelProvider
    supports_structured_output = True
    supports_tool_calling = False
    supports_web_grounding = False

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    @abstractmethod
    def api_key(self) -> str | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_model(self) -> str:
        raise NotImplementedError

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.provider,
            configured=self.configured,
            default_model=self.default_model,
            supports_structured_output=self.supports_structured_output,
            supports_tool_calling=self.supports_tool_calling,
            supports_web_grounding=self.supports_web_grounding,
        )

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        if not self.configured:
            raise ProviderInvocationError(f"Provider '{self.provider.value}' is not configured.")

        payload, url, headers = self._request_config(request)
        started = perf_counter()
        try:
            with httpx.Client(timeout=self._settings.model_timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - networked path
            raise ProviderInvocationError(str(exc)) from exc

        raw_payload = response.json()
        latency_ms = int((perf_counter() - started) * 1000)
        return ProviderResponse(
            provider=self.provider,
            model=request.model,
            content=self._extract_text(raw_payload),
            latency_ms=latency_ms,
            finish_reason=self._extract_finish_reason(raw_payload),
            raw_payload=raw_payload if isinstance(raw_payload, dict) else {"response": raw_payload},
        )

    def _messages_with_schema(self, request: ProviderRequest) -> list[ProviderMessage]:
        messages = [message.model_copy(deep=True) for message in request.messages]
        if request.structured_output is None:
            return messages

        schema_text = json.dumps(request.structured_output.json_schema, ensure_ascii=True)
        instruction = (
            "Return only a valid JSON object that matches this schema exactly.\n"
            f"{schema_text}"
        )
        if messages and messages[0].role == "system":
            messages[0] = messages[0].model_copy(
                update={"content": f"{messages[0].content}\n\n{instruction}".strip()}
            )
            return messages
        return [ProviderMessage(role="system", content=instruction), *messages]

    @abstractmethod
    def _request_config(
        self,
        request: ProviderRequest,
    ) -> tuple[dict[str, Any], str, dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def _extract_text(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError

    def _extract_finish_reason(self, payload: dict[str, Any]) -> str | None:
        return None
