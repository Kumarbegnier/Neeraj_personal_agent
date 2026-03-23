from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    app_name: str = Field(default_factory=lambda: os.getenv("APP_NAME", "Neeraj AI OS"))
    app_host: str = Field(default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"))
    app_port: int = Field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))

    require_auth: bool = Field(default_factory=lambda: _env_bool("REQUIRE_AUTH", False))
    dev_auth_bypass: bool = Field(default_factory=lambda: _env_bool("DEV_AUTH_BYPASS", True))
    enable_debug_trace: bool = Field(default_factory=lambda: _env_bool("ENABLE_DEBUG_TRACE", True))

    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_chat_model: str = Field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_responses_model: str = Field(default_factory=lambda: os.getenv("OPENAI_RESPONSES_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")))
    openai_embedding_model: str = Field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")))
    openai_embedding_dimensions: int = Field(default_factory=lambda: int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")))
    use_openai_agents_sdk: bool = Field(default_factory=lambda: _env_bool("USE_OPENAI_AGENTS_SDK", True))
    claude_api_key: str | None = Field(default_factory=lambda: os.getenv("CLAUDE_API_KEY", os.getenv("ANTHROPIC_API_KEY")))
    claude_model: str = Field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest"))
    gemini_api_key: str | None = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY")))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
    deepseek_api_key: str | None = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    deepseek_model: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"))
    model_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("MODEL_TIMEOUT_SECONDS", "45")))

    mongodb_uri: str | None = Field(default_factory=lambda: os.getenv("MONGODB_URI"))
    mongodb_db_name: str = Field(default_factory=lambda: os.getenv("MONGODB_DB_NAME", "neeraj_ai_os"))
    mongodb_episodic_collection: str = Field(default_factory=lambda: os.getenv("MONGODB_EPISODIC_COLLECTION", "episodic_memory"))
    mongodb_task_collection: str = Field(default_factory=lambda: os.getenv("MONGODB_TASK_COLLECTION", "task_logs"))

    semantic_backend: str = Field(default_factory=lambda: os.getenv("SEMANTIC_BACKEND", "in_memory_faiss_compatible"))
    semantic_top_k: int = Field(default_factory=lambda: int(os.getenv("SEMANTIC_TOP_K", "6")))

    playwright_browser: str = Field(default_factory=lambda: os.getenv("PLAYWRIGHT_BROWSER", "chromium"))
    playwright_headless: bool = Field(default_factory=lambda: _env_bool("PLAYWRIGHT_HEADLESS", True))

    audit_log_file: str = Field(default_factory=lambda: os.getenv("AUDIT_LOG_FILE", "audit.log"))
    allow_tool_side_effects: bool = Field(default_factory=lambda: _env_bool("ALLOW_TOOL_SIDE_EFFECTS", False))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
