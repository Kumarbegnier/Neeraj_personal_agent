from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable


def mongo_enabled() -> bool:
    return bool(os.getenv("MONGODB_URI"))


def get_collection():
    """Return pymongo collection using env vars.

    Env:
    - MONGODB_URI
    - MONGODB_DB (default: automation1)
    - MONGODB_COLLECTION (default: spaces)
    """

    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set")

    try:
        from pymongo import MongoClient  # type: ignore
    except Exception as exc:
        raise RuntimeError("pymongo is not installed. Add it to requirements.txt") from exc

    db_name = os.getenv("MONGODB_DB", "automation1")
    coll_name = os.getenv("MONGODB_COLLECTION", "spaces")

    client = MongoClient(uri)
    return client[db_name][coll_name]


def upsert_space_records(records: Iterable[dict]) -> int:
    """Upsert records into Mongo (by _id). Returns number of attempted upserts."""

    coll = get_collection()
    count = 0
    now = datetime.now(timezone.utc)

    for rec in records:
        if "_id" not in rec:
            continue
        rec = dict(rec)
        rec.setdefault("updated_at", now)
        rec.setdefault("created_at", now)
        coll.update_one({"_id": rec["_id"]}, {"$set": rec, "$setOnInsert": {"created_at": now}}, upsert=True)
        count += 1

    return count
