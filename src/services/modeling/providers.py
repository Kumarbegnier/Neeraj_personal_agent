from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel

from src.core.config import Settings

from .types import ModelProvider


class ProviderInvocationError(RuntimeError):
    pass


class BaseProviderAdapter(ABC):
    provider: ModelProvider

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

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.provider.value,
            "configured": self.configured,
            "default_model": self.default_model,
            "supports_structured_output": True,
        }

    def generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_type: type[BaseModel],
    ) -> BaseModel:
        if not self.configured:
            raise ProviderInvocationError(f"Provider '{self.provider.value}' is not configured.")

        prompt = self._json_prompt(system_prompt, user_prompt, output_type)
        payload, url, headers = self._request_config(model=model, prompt=prompt)
        try:
            with httpx.Client(timeout=self._settings.model_timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - networked path
            raise ProviderInvocationError(str(exc)) from exc

        content = self._extract_text(response.json())
        parsed = self._extract_json(content)
        return output_type.model_validate(parsed)

    def _json_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        output_type: type[BaseModel],
    ) -> str:
        schema = json.dumps(output_type.model_json_schema(), ensure_ascii=True)
        return (
            f"{system_prompt}\n\n"
            "Return only a valid JSON object that matches this schema exactly.\n"
            f"{schema}\n\n"
            f"{user_prompt}"
        )

    def _extract_json(self, content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ProviderInvocationError("The provider did not return valid JSON.")
            return json.loads(content[start : end + 1])

    @abstractmethod
    def _request_config(
        self,
        *,
        model: str,
        prompt: str,
    ) -> tuple[dict[str, Any], str, dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def _extract_text(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError


class OpenAICompatibleProviderAdapter(BaseProviderAdapter):
    base_url: str

    def _request_config(
        self,
        *,
        model: str,
        prompt: str,
    ) -> tuple[dict[str, Any], str, dict[str, str]]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a structured-output model. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return payload, f"{self.base_url}/chat/completions", headers

    def _extract_text(self, payload: dict[str, Any]) -> str:
        return str(payload["choices"][0]["message"]["content"])


class OpenAIProviderAdapter(OpenAICompatibleProviderAdapter):
    provider = ModelProvider.OPENAI
    base_url = "https://api.openai.com/v1"

    @property
    def api_key(self) -> str | None:
        return self._settings.openai_api_key

    @property
    def default_model(self) -> str:
        return self._settings.openai_responses_model


class DeepSeekProviderAdapter(OpenAICompatibleProviderAdapter):
    provider = ModelProvider.DEEPSEEK
    base_url = "https://api.deepseek.com/v1"

    @property
    def api_key(self) -> str | None:
        return self._settings.deepseek_api_key

    @property
    def default_model(self) -> str:
        return self._settings.deepseek_model


class ClaudeProviderAdapter(BaseProviderAdapter):
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
        *,
        model: str,
        prompt: str,
    ) -> tuple[dict[str, Any], str, dict[str, str]]:
        payload = {
            "model": model,
            "max_tokens": 2048,
            "system": "You are a structured-output model. Return JSON only.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        }
        headers = {
            "x-api-key": str(self.api_key),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        return payload, self.base_url, headers

    def _extract_text(self, payload: dict[str, Any]) -> str:
        content = payload.get("content", [])
        if not content:
            raise ProviderInvocationError("Claude did not return any content.")
        first = content[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
        return str(first)


class GeminiProviderAdapter(BaseProviderAdapter):
    provider = ModelProvider.GEMINI
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def api_key(self) -> str | None:
        return self._settings.gemini_api_key

    @property
    def default_model(self) -> str:
        return self._settings.gemini_model

    def _request_config(
        self,
        *,
        model: str,
        prompt: str,
    ) -> tuple[dict[str, Any], str, dict[str, str]]:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
        return payload, url, {"Content-Type": "application/json"}

    def _extract_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ProviderInvocationError("Gemini did not return any candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise ProviderInvocationError("Gemini did not return any content parts.")
        return str(parts[0].get("text", ""))
