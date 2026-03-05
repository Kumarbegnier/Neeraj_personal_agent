from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# only for USER


@dataclass(frozen=True)
class SpaceRecord:
    record_id: str
    city: str
    address: str
    gst_price: int | None
    br_price: int | None
    mail_price: int | None


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def load_spaces_from_json(path: str | Path) -> list[SpaceRecord]:
    """Load space records from a JSON file shaped like: { city: [ {address, ...}, ...] }"""

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object mapping city -> list")

    records: list[SpaceRecord] = []

    for city, items in data.items():
        if not isinstance(city, str):
            continue
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            address = item.get("address")
            # tolerate the known typo without mutating source
            if not address and "addreess" in item:
                address = item.get("addreess")

            if not isinstance(address, str) or not address.strip():
                continue

            def _int_or_none(v: Any) -> int | None:
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    return int(v)
                if isinstance(v, str) and v.strip().isdigit():
                    return int(v.strip())
                return None

            gst_price = _int_or_none(item.get("gst_price"))
            br_price = _int_or_none(item.get("br_price"))
            mail_price = _int_or_none(item.get("mail_price"))

            record_id = _stable_id(
                city.strip().lower(),
                address.strip().lower(),
                str(gst_price or ""),
                str(br_price or ""),
                str(mail_price or ""),
            )

            records.append(
                SpaceRecord(
                    record_id=record_id,
                    city=city.strip(),
                    address=address.strip(),
                    gst_price=gst_price,
                    br_price=br_price,
                    mail_price=mail_price,
                )
            )

    return records


def records_to_texts(records: Iterable[SpaceRecord]) -> list[str]:
    texts: list[str] = []
    for r in records:
        # Keep it compact; retrieval should be able to match by city + address + prices.
        parts = [f"City: {r.city}", f"Address: {r.address}"]
        if r.gst_price is not None:
            parts.append(f"GST price: {r.gst_price}")
        if r.br_price is not None:
            parts.append(f"BR price: {r.br_price}")
        if r.mail_price is not None:
            parts.append(f"Mail price: {r.mail_price}")
        texts.append("\n".join(parts))
    return texts
