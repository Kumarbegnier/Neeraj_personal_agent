from __future__ import annotations


def validate_user_message(message: str) -> str:
    normalized = " ".join(message.split())
    if not normalized:
        raise ValueError("Message must not be empty.")
    return normalized
