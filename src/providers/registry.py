from __future__ import annotations

from src.core.config import Settings
from src.schemas.routing import ModelProvider

from .base import BaseProviderClient
from .claude import ClaudeProviderClient
from .deepseek import DeepSeekProviderClient
from .gemini import GeminiProviderClient
from .openai import OpenAIProviderClient


class ProviderRegistry:
    def __init__(self, settings: Settings) -> None:
        self._providers: dict[ModelProvider, BaseProviderClient] = {
            ModelProvider.OPENAI: OpenAIProviderClient(settings),
            ModelProvider.CLAUDE: ClaudeProviderClient(settings),
            ModelProvider.GEMINI: GeminiProviderClient(settings),
            ModelProvider.DEEPSEEK: DeepSeekProviderClient(settings),
        }

    def get(self, provider: ModelProvider) -> BaseProviderClient:
        return self._providers[provider]

    def all(self) -> dict[ModelProvider, BaseProviderClient]:
        return dict(self._providers)

    def health(self) -> dict[str, dict[str, object]]:
        return {
            provider.value: client.health().model_dump(mode="json")
            for provider, client in self._providers.items()
        }
