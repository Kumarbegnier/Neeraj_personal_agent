from __future__ import annotations

from typing import Any

from src.schemas.provider import ProviderRequest, StructuredOutputSchema
from src.schemas.routing import ModelProvider

from .base import BaseProviderClient, ProviderInvocationError


class OpenAICompatibleProviderClient(BaseProviderClient):
    base_url: str

    def _request_config(
        self,
        request: ProviderRequest,
    ) -> tuple[dict[str, Any], str, dict[str, str]]:
        messages = self._messages_with_schema(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.structured_output is not None:
            payload["response_format"] = self._structured_response_format(request.structured_output)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return payload, f"{self.base_url}/chat/completions", headers

    def _extract_text(self, payload: dict[str, Any]) -> str:
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderInvocationError("The provider did not return a chat completion payload.") from exc

    def _extract_finish_reason(self, payload: dict[str, Any]) -> str | None:
        choices = payload.get("choices", [])
        if not choices:
            return None
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            return first_choice.get("finish_reason")
        return None

    def _structured_response_format(self, schema: StructuredOutputSchema) -> dict[str, Any]:
        return {"type": "json_object"}


class OpenAIProviderClient(OpenAICompatibleProviderClient):
    provider = ModelProvider.OPENAI
    base_url = "https://api.openai.com/v1"
    supports_tool_calling = True

    @property
    def api_key(self) -> str | None:
        return self._settings.openai_api_key

    @property
    def default_model(self) -> str:
        return self._settings.openai_responses_model

    def _structured_response_format(self, schema: StructuredOutputSchema) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.name,
                "schema": schema.json_schema,
                "strict": schema.strict,
            },
        }
