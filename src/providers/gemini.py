from __future__ import annotations

from typing import Any

from src.schemas.provider import ProviderRequest
from src.schemas.routing import ModelProvider

from .base import BaseProviderClient, ProviderInvocationError


class GeminiProviderClient(BaseProviderClient):
    provider = ModelProvider.GEMINI
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    supports_web_grounding = True

    @property
    def api_key(self) -> str | None:
        return self._settings.gemini_api_key

    @property
    def default_model(self) -> str:
        return self._settings.gemini_model

    def _request_config(
        self,
        request: ProviderRequest,
    ) -> tuple[dict[str, Any], str, dict[str, str]]:
        messages = self._messages_with_schema(request)
        system_messages = [message.content for message in messages if message.role == "system"]
        contents = [
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            }
            for message in messages
            if message.role != "system"
        ]
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Respond to the request."}]}]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        if system_messages:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_messages)}],
            }
        if request.structured_output is not None:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if request.web_grounded:
            payload["tools"] = [{"google_search": {}}]

        url = f"{self.base_url}/{request.model}:generateContent?key={self.api_key}"
        return payload, url, {"Content-Type": "application/json"}

    def _extract_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ProviderInvocationError("Gemini did not return any candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [str(part.get("text", "")) for part in parts if isinstance(part, dict)]
        if not texts:
            raise ProviderInvocationError("Gemini did not return any text parts.")
        return "\n".join(texts).strip()

    def _extract_finish_reason(self, payload: dict[str, Any]) -> str | None:
        candidates = payload.get("candidates", [])
        if not candidates:
            return None
        first_candidate = candidates[0]
        if isinstance(first_candidate, dict):
            return first_candidate.get("finishReason")
        return None
