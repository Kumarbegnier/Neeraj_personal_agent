from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.session_store import get_session_messages, save_message, clear_session_messages

logger = logging.getLogger(__name__)


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """LangChain history stored in SQLite via app.session_store (keyed by session_id)."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:
        rows = get_session_messages(self.session_id)
        logger.info("Loaded %s history messages from SQLite for session_id=%s", len(rows), self.session_id)
        out: list[BaseMessage] = []

        for r in rows:
            role = (r.role or "").lower()
            content = (r.content or "").strip()
            if not content:
                continue

            if role in ("user", "human"):
                out.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                out.append(AIMessage(content=content))
            elif role == "system":
                out.append(SystemMessage(content=content))
            else:
                # Fallback: preserve content as a human message
                out.append(HumanMessage(content=content))

        return out

    def add_message(self, message: BaseMessage) -> None:
        msg_type = (getattr(message, "type", "") or "").lower()

        if msg_type in ("human", "user"):
            role = "user"
        elif msg_type in ("ai", "assistant"):
            role = "ai"
        elif msg_type == "system":
            role = "system"
        else:
            role = "user"

        content = message.content if isinstance(message.content, str) else str(message.content)
        save_message(self.session_id, role, content)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        clear_session_messages(self.session_id)


@lru_cache(maxsize=2048)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLiteChatMessageHistory(session_id=session_id)