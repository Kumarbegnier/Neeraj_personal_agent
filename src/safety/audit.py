from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.core.config import get_settings
from src.schemas.catalog import AuditEvent

logger = logging.getLogger(__name__)


class AuditService:
    """Simple audit sink that logs to stdout and a local file for Windows-friendly development."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._path = Path(self.settings.audit_log_file)
        self._events: list[AuditEvent] = []

    def record(self, event: str, payload: dict[str, Any]) -> None:
        audit_event = AuditEvent(event=event, payload=payload)
        self._events.append(audit_event)
        self._events = self._events[-500:]
        logger.info("audit_event=%s payload=%s", event, payload)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit_event.model_dump(mode="json"), default=str) + "\n")
        except OSError:
            logger.warning("Failed to write audit event to %s", self._path)

    def recent(self, limit: int = 100) -> list[AuditEvent]:
        if self._events:
            return [event.model_copy(deep=True) for event in self._events[-limit:]]
        if not self._path.exists():
            return []

        events: list[AuditEvent] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()[-limit:]
        except OSError:
            return []

        for line in lines:
            try:
                raw = json.loads(line)
                if "recorded_at" not in raw:
                    raw = {
                        "event": raw.get("event", "unknown"),
                        "payload": raw.get("payload", {}),
                    }
                events.append(AuditEvent.model_validate(raw))
            except Exception:
                continue
        return events

    def health(self) -> dict[str, Any]:
        return {
            "audit_log_file": str(self._path),
            "exists": self._path.exists(),
            "recent_event_count": len(self._events),
        }
