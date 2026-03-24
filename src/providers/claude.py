from __future__ import annotations

from typing import Any

from src.schemas.provider import ProviderRequest
from src.schemas.routing import ModelProvider

from .base import BaseProviderClient, ProviderInvocationError


class ClaudeProviderClient(BaseProviderClient):
    provider = ModelProvider.CLAUDE
    base_url = "https://api.anthropic.com/v1/messages"

    @property
    def api_key(self) -> str | None:
        return self._settings.claude_api_key

    @property
    def default_model(self) -> str:
        return self._settings.claude_model

    def _request_config(
        self,
        request: ProviderRequest,
    ) -> tuple[dict[str, Any], str, dict[str, str]]:
        messages = self._messages_with_schema(request)
        system_prompt = "\n\n".join(message.content for message in messages if message.role == "system").strip()
        conversation = [
            {
                "role": "assistant" if message.role == "assistant" else "user",
                "content": [{"type": "text", "text": message.content}],
            }
            for message in messages
            if message.role != "system"
        ]
        if not conversation:
            conversation = [{"role": "user", "content": [{"type": "text", "text": "Respond to the request."}]}]

        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": conversation,
        }
        if system_prompt:
            payload["system"] = system_prompt
        headers = {
            "x-api-key": str(self.api_key),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        return payload, self.base_url, headers

    def _extract_text(self, payload: dict[str, Any]) -> str:
        content = payload.get("content", [])
        texts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"]
        if not texts:
            raise ProviderInvocationError("Claude did not return any text content.")
        return "\n".join(texts).strip()

    def _extract_finish_reason(self, payload: dict[str, Any]) -> str | None:
        return payload.get("stop_reason")
