from __future__ import annotations

import re
from typing import Any


BOOKING_INTENT_RE = re.compile(
    r"(book|booking|looking\s+for|find|search|workspace|space|virtual|coworking|meeting)",
    re.IGNORECASE,
)

SPACE_OPTIONS = [
    "Virtual Office",
    "Coworking Desk",
    "Dedicated Desk",
    "Private Cabin",
    "Meeting Room",
]

DURATION_OPTIONS = ["Daily", "Weekly", "Monthly", "Long-term (3+ months)"]

_STATE: dict[str, dict[str, Any]] = {}


def _ask_location() -> str:
    return (
        "Please share your location in this format:\n"
        "- Country\n"
        "- State\n"
        "- City"
    )


def _ask_type() -> str:
    return (
        "Type of Space:\n"
        "- Virtual Office\n"
        "- Coworking Desk\n"
        "- Dedicated Desk\n"
        "- Private Cabin\n"
        "- Meeting Room"
    )


def _ask_people() -> str:
    return "Number of People?"


def _ask_budget() -> str:
    return "Budget / Price Range?"


def _ask_duration() -> str:
    return "Duration?\n- Daily\n- Weekly\n- Monthly\n- Long-term (3+ months)"


def _normalize_space_type(text: str) -> str:
    t = (text or "").strip().lower()
    if "virtual" in t:
        return "Virtual Office"
    if "cowork" in t:
        return "Coworking Desk"
    if "dedicated" in t:
        return "Dedicated Desk"
    if "private" in t or "cabin" in t:
        return "Private Cabin"
    if "meeting" in t:
        return "Meeting Room"
    return ""


def _normalize_duration(text: str) -> str:
    t = (text or "").strip().lower()
    if "daily" in t:
        return "Daily"
    if "weekly" in t:
        return "Weekly"
    if "monthly" in t:
        return "Monthly"
    if "long" in t or "3" in t:
        return "Long-term (3+ months)"
    return ""


def _extract_people(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"\b\d+\b", t)
    if m:
        return m.group(0)
    return t


def process_booking_discovery_flow(query: str, *, session_id: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""

    st = _STATE.get(session_id)

    if st and st.get("active"):
        step = st.get("step")
        data = st.setdefault("data", {})

        if step == "location":
            data["location"] = q
            st["step"] = "space_type"
            return _ask_type()

        if step == "space_type":
            space_type = _normalize_space_type(q)
            if not space_type:
                return _ask_type()
            data["space_type"] = space_type
            st["step"] = "people"
            return _ask_people()

        if step == "people":
            data["people"] = _extract_people(q)
            st["step"] = "budget"
            return _ask_budget()

        if step == "budget":
            data["budget"] = q
            st["step"] = "duration"
            return _ask_duration()

        if step == "duration":
            duration = _normalize_duration(q)
            data["duration"] = duration or q
            st["step"] = "done"
            st["active"] = False
            return (
                "Showing Results:\n"
                f"- Location: {data.get('location', 'N/A')}\n"
                f"- Type of Space: {data.get('space_type', 'N/A')}\n"
                f"- Number of People: {data.get('people', 'N/A')}\n"
                f"- Budget / Price Range: {data.get('budget', 'N/A')}\n"
                f"- Duration: {data.get('duration', 'N/A')}\n"
                "Would you like me to show recommended options now?"
            )

    if BOOKING_INTENT_RE.search(q):
        _STATE[session_id] = {"active": True, "step": "location", "data": {}}
        return _ask_location()

    return ""

