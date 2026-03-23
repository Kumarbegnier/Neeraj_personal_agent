from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from src.core.config import get_settings


def _dedupe(items: list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return tuple(ordered)


@dataclass(frozen=True)
class FrontendConfig:
    app_title: str
    backend_url: str
    request_timeout_seconds: float
    model_options: tuple[str, ...]


@lru_cache(maxsize=1)
def get_frontend_config() -> FrontendConfig:
    settings = get_settings()
    backend_url = os.getenv("NEERAJ_API_URL", f"http://127.0.0.1:{settings.app_port}")
    request_timeout_seconds = float(os.getenv("NEERAJ_API_TIMEOUT_SECONDS", "60"))
    model_options = _dedupe(
        [
            settings.openai_responses_model,
            settings.openai_chat_model,
            settings.claude_model,
            settings.gemini_model,
            settings.deepseek_model,
            "gpt-4o-mini",
            "claude-3-5-sonnet-latest",
            "gemini-1.5-pro",
            "deepseek-reasoner",
        ]
    )
    return FrontendConfig(
        app_title=settings.app_name,
        backend_url=backend_url,
        request_timeout_seconds=request_timeout_seconds,
        model_options=model_options,
    )
