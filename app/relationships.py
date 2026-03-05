from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


def _normalize_model_name(value: str) -> str:
    key = (value or "").strip().lower().replace("_", "").replace(" ", "")
    if key in {"virtualoffice", "virtualoffices"}:
        return "virtualoffices"
    if key in {"coworkingspace", "coworkingspaces"}:
        return "coworkingspaces"
    if key in {"meetingroom", "meetingrooms"}:
        return "meetingrooms"
    return ""


def _oid_from_value(value: Any) -> str | None:
    if isinstance(value, dict):
        oid = value.get("$oid")
        if isinstance(oid, str) and oid:
            return oid
    if isinstance(value, str) and len(value) == 24:
        return value
    return None


def _iter_oid_fields(value: Any, path: str = ""):
    oid = _oid_from_value(value)
    if oid is not None:
        yield path, oid
        return

    if isinstance(value, dict):
        for k, v in value.items():
            next_path = f"{path}.{k}" if path else k
            yield from _iter_oid_fields(v, next_path)
        return

    if isinstance(value, list):
        for v in value:
            next_path = f"{path}[]" if path else "[]"
            yield from _iter_oid_fields(v, next_path)


def _load_collections_from_json_dir(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not path.exists() or not path.is_dir():
        return out

    for fp in sorted(path.glob("test.*.json")):
        name = fp.name[len("test.") : -len(".json")]
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, list):
            docs = [d for d in raw if isinstance(d, dict)]
            out[name] = docs
    return out


def _id_sets(collections: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name, docs in collections.items():
        ids: set[str] = set()
        for d in docs:
            oid = _oid_from_value(d.get("_id"))
            if oid:
                ids.add(oid)
        out[name] = ids
    return out


def _infer_simple_edges(
    collections: dict[str, list[dict[str, Any]]], ids_by_collection: dict[str, set[str]]
) -> list[str]:
    lines: list[str] = []
    for source, docs in collections.items():
        field_values: dict[str, list[str]] = defaultdict(list)
        for d in docs:
            for path, oid in _iter_oid_fields(d):
                if path == "_id":
                    continue
                field_values[path].append(oid)

        for field, values in sorted(field_values.items()):
            if len(values) < 3:
                continue
            best_target = ""
            best_matches = 0
            for target, target_ids in ids_by_collection.items():
                matches = sum(1 for v in values if v in target_ids)
                if matches > best_matches:
                    best_matches = matches
                    best_target = target

            if not best_target or best_target == source:
                continue
            ratio = best_matches / len(values)
            if ratio < 0.8:
                continue
            lines.append(f"- `{source}.{field}` -> `{best_target}._id` ({best_matches}/{len(values)})")
    return lines


def _infer_polymorphic_space_edges(
    collections: dict[str, list[dict[str, Any]]], ids_by_collection: dict[str, set[str]]
) -> list[str]:
    lines: list[str] = []
    rules = [
        ("bookings", "spaceId", "type"),
        ("payments", "space", "spaceModel"),
        ("reviews", "space", "spaceModel"),
    ]

    for source, id_field, discriminator in rules:
        docs = collections.get(source) or []
        if not docs:
            continue
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for d in docs:
            oid = _oid_from_value(d.get(id_field))
            model = _normalize_model_name(str(d.get(discriminator) or ""))
            if not oid or not model:
                continue
            grouped[(id_field, model)].append(oid)

        for (field, target), values in sorted(grouped.items()):
            target_ids = ids_by_collection.get(target, set())
            if not target_ids:
                continue
            matches = sum(1 for v in values if v in target_ids)
            if matches == 0:
                continue
            lines.append(
                f"- `{source}.{field}` + `{source}.{discriminator}={target}` -> `{target}._id` ({matches}/{len(values)})"
            )

    # Known string FK in sample dump.
    seat_docs = collections.get("seatbookings") or []
    payment_ids = ids_by_collection.get("payments", set())
    if seat_docs and payment_ids:
        values = [str(d.get("paymentId")) for d in seat_docs if d.get("paymentId")]
        if values:
            matches = sum(1 for v in values if v in payment_ids)
            if matches:
                lines.append(f"- `seatbookings.paymentId` -> `payments._id` ({matches}/{len(values)})")

    return lines


@lru_cache(maxsize=1)
def get_relationship_context() -> str:
    json_dir = (os.getenv("RELATIONSHIP_JSON_DIR") or "").strip()
    if not json_dir:
        return ""

    collections = _load_collections_from_json_dir(Path(json_dir))
    if not collections:
        return ""

    ids_by_collection = _id_sets(collections)
    simple_edges = _infer_simple_edges(collections, ids_by_collection)
    poly_edges = _infer_polymorphic_space_edges(collections, ids_by_collection)

    lines: list[str] = []
    lines.append(f"Relationship source: {json_dir}")
    lines.append(f"Collections loaded: {', '.join(sorted(collections.keys()))}")
    if simple_edges:
        lines.append("Direct relationships:")
        lines.extend(simple_edges)
    if poly_edges:
        lines.append("Polymorphic relationships:")
        lines.extend(poly_edges)

    return "\n".join(lines).strip()
