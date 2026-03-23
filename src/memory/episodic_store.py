from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from src.core.config import get_settings
from src.runtime.models import ConversationTurn

try:  # pragma: no cover - optional dependency
    from pymongo import MongoClient
except Exception:  # pragma: no cover - optional dependency
    MongoClient = None  # type: ignore[assignment]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EpisodicStore:
    """Durable episodic memory and task log adapter with an in-memory fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._fallback_turns: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._fallback_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._client = None
        self._collection = None
        self._task_collection = None
        self._connect()

    def _connect(self) -> None:
        if MongoClient is None or not self.settings.mongodb_uri:
            return
        try:  # pragma: no cover - depends on external service
            self._client = MongoClient(self.settings.mongodb_uri, serverSelectionTimeoutMS=1000)
            database = self._client[self.settings.mongodb_db_name]
            self._collection = database[self.settings.mongodb_episodic_collection]
            self._task_collection = database[self.settings.mongodb_task_collection]
            self._client.admin.command("ping")
        except Exception:
            self._client = None
            self._collection = None
            self._task_collection = None

    def health(self) -> dict[str, Any]:
        return {
            "backend": "mongodb" if self._collection is not None else "in_memory_fallback",
            "connected": self._collection is not None,
        }

    def _key(self, user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    def save_interaction(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        request_text: str,
        response_text: str,
        metadata: dict[str, Any] | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> str:
        document = {
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
            "request_text": request_text,
            "response_text": response_text,
            "metadata": metadata or {},
            "trace": trace or [],
            "created_at": _utc_now(),
        }
        if self._collection is not None:  # pragma: no cover - depends on external service
            self._collection.insert_one(document)
        else:
            self._fallback_turns[self._key(user_id, session_id)].append(document)
        return request_id

    def store_task_outcome(
        self,
        *,
        user_id: str,
        session_id: str,
        task_name: str,
        outcome: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        task_id = f"task-{user_id}-{session_id}-{len(self._fallback_tasks[self._key(user_id, session_id)]) + 1}"
        document = {
            "task_id": task_id,
            "user_id": user_id,
            "session_id": session_id,
            "task_name": task_name,
            "outcome": outcome,
            "metadata": metadata or {},
            "created_at": _utc_now(),
        }
        if self._task_collection is not None:  # pragma: no cover - depends on external service
            self._task_collection.insert_one(document)
        else:
            self._fallback_tasks[self._key(user_id, session_id)].append(document)
        return task_id

    def load_recent_context(
        self,
        user_id: str,
        session_id: str,
        limit: int = 6,
    ) -> list[ConversationTurn]:
        key = self._key(user_id, session_id)
        if self._collection is not None:  # pragma: no cover - depends on external service
            documents = list(
                self._collection.find({"user_id": user_id, "session_id": session_id})
                .sort("created_at", -1)
                .limit(limit)
            )
            turns: list[ConversationTurn] = []
            for document in reversed(documents):
                turns.append(
                    ConversationTurn(
                        role="user",
                        content=document.get("request_text", ""),
                        timestamp=document.get("created_at", _utc_now()),
                        metadata=document.get("metadata", {}),
                    )
                )
                turns.append(
                    ConversationTurn(
                        role="assistant",
                        content=document.get("response_text", ""),
                        timestamp=document.get("created_at", _utc_now()),
                        metadata=document.get("metadata", {}),
                    )
                )
            return turns[-limit:]

        documents = self._fallback_turns[key][-limit:]
        turns = []
        for document in documents:
            turns.append(
                ConversationTurn(
                    role="user",
                    content=document["request_text"],
                    timestamp=document["created_at"],
                    metadata=document["metadata"],
                )
            )
            turns.append(
                ConversationTurn(
                    role="assistant",
                    content=document["response_text"],
                    timestamp=document["created_at"],
                    metadata=document["metadata"],
                )
            )
        return turns[-limit:]

    def recent(
        self,
        user_id: str,
        session_id: str,
        limit: int = 6,
    ) -> list[ConversationTurn]:
        return self.load_recent_context(user_id=user_id, session_id=session_id, limit=limit)
