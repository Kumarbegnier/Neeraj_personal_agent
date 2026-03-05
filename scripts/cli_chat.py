from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _load_dotenv_simple(path: Path) -> None:
    """Minimal .env loader (no external dependency)."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ.setdefault(key, value)


def _normalize_token(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    parts = value.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return value


def main() -> int:
    _load_dotenv_simple(BACKEND_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Terminal chat client for the FastAPI /chat endpoint")
    parser.add_argument(
        "--base-url",
        default=os.getenv("CLI_BASE_URL") or "http://127.0.0.1:8000",
        help="Base URL of the backend (default: CLI_BASE_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("CLI_JWT") or "",
        help="JWT token (or 'Bearer <token>'). If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--conversation-id",
        default=os.getenv("CLI_CONVERSATION_ID") or "default",
        help="Conversation id for server-side session grouping.",
    )
    parser.add_argument(
        "--once",
        default=None,
        help="Send a single query and exit (non-interactive).",
    )
    args = parser.parse_args()

    base_url = str(args.base_url).rstrip("/")
    url = f"{base_url}/chat"

    print(f"Target: {url}")
    print("Commands: exit | quit | :token | :conv <id> | :session <id> | :session")

    token = _normalize_token(str(args.token))
    if not token:
        token = _normalize_token(input("Paste JWT (or 'Bearer <jwt>'): ").strip())

    conversation_id = str(args.conversation_id).strip() or "default"
    session_id: str | None = os.getenv("CLI_SESSION_ID") or None

    def _send(query: str) -> int:
        nonlocal session_id
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"query": query, "conversation_id": conversation_id}
        if session_id:
            payload["session_id"] = session_id
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
        except Exception as exc:
            print(f"request_failed: {exc.__class__.__name__}: {exc}")
            return 1

        if resp.status_code != 200:
            print(f"HTTP {resp.status_code}: {resp.text}")
            return 1

        try:
            data = resp.json()
        except Exception:
            print(resp.text)
            return 1

        returned_session_id = (data.get("session_id") or "").strip()
        if returned_session_id:
            session_id = returned_session_id

        print(data.get("reply", ""))
        return 0

    if args.once is not None:
        return _send(str(args.once))

    while True:
        query = input(f"[conv={conversation_id}]> ").strip()
        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            return 0

        if query.startswith(":token"):
            token = _normalize_token(input("Paste JWT (or 'Bearer <jwt>'): ").strip())
            continue

        if query.startswith(":conv"):
            parts = query.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                conversation_id = parts[1].strip()
            else:
                print("usage: :conv <conversation_id>")
            continue

        if query.startswith(":session"):
            parts = query.split(maxsplit=1)
            if len(parts) == 1:
                print(session_id or "(no session_id yet)")
            elif parts[1].strip():
                session_id = parts[1].strip()
                print(f"session_id set: {session_id}")
            else:
                print("usage: :session <session_id>")
            continue

        _send(query)


if __name__ == "__main__":
    raise SystemExit(main())
