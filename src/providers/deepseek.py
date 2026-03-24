from __future__ import annotations

from src.schemas.routing import ModelProvider

from .openai import OpenAICompatibleProviderClient


class DeepSeekProviderClient(OpenAICompatibleProviderClient):
    provider = ModelProvider.DEEPSEEK
    base_url = "https://api.deepseek.com/v1"

    @property
    def api_key(self) -> str | None:
        return self._settings.deepseek_api_key

    @property
    def default_model(self) -> str:
        return self._settings.deepseek_model
