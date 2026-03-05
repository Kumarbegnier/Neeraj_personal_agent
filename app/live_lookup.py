from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
OID_RE = re.compile(r"\b[a-fA-F0-9]{24}\b")
DETAIL_INTENT_RE = re.compile(
    r"(my\s+detail|account|profile|my\s+info|show\s+me\s+all\s+details|meri\s+detail|details?)",
    re.IGNORECASE,
)
VIRTUAL_SPACE_INTENT_RE = re.compile(
    r"(virtual\s*office|virtual\s*space|business\s*address|gst\s*address|virtual|office|business)",
    re.IGNORECASE,
)

SENSITIVE_KEYS = {
    "password",
    "refreshTokens",
    "razorpaySignature",
    "emailVerificationOTP",
    "otp",
    "emailVerificationOTPAttempts",
    "__v",
}

_PENDING_IDENTITY: dict[str, dict[str, str]] = {}


def _extract_terms(query: str) -> tuple[list[str], list[str]]:
    emails = sorted(set(EMAIL_RE.findall(query or "")))
    oids = sorted(set(v.lower() for v in OID_RE.findall(query or "")))
    return emails, oids


def _extract_password_input(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    lowered = q.lower()
    prefixes = ["password:", "pass:", "pwd:"]
    for p in prefixes:
        if lowered.startswith(p):
            q = q[len(p) :].strip()
            break
    # Normalize accidental spaces/newlines users paste from split hash strings.
    q = q.replace(" ", "").replace("\n", "").replace("\r", "").strip()
    return q


def _normalize_bcrypt_like(value: str) -> str:
    v = (value or "").strip().replace(" ", "").replace("\n", "").replace("\r", "")
    if not v:
        return ""
    # Accept pasted variant like: 2b10$abc... -> $2b$10$abc...
    m = re.match(r"^2([aby])(\d\d)\$(.+)$", v)
    if m:
        return f"$2{m.group(1)}${m.group(2)}${m.group(3)}"
    # Accept pasted variant like: 2b$10$abc... -> $2b$10$abc...
    m = re.match(r"^2([aby])\$(\d\d)\$(.+)$", v)
    if m:
        return f"$2{m.group(1)}${m.group(2)}${m.group(3)}"
    return v


@lru_cache(maxsize=1)
def _get_client():
    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None
    try:
        from pymongo import MongoClient  # type: ignore
        return MongoClient(uri, serverSelectionTimeoutMS=3500)
    except Exception:
        return None


def _db_candidates() -> list[str]:
    env_list = (os.getenv("MONGODB_LOOKUP_DBS") or "").strip()
    names: list[str] = []
    if env_list:
        names.extend([x.strip() for x in env_list.split(",") if x.strip()])
    fallback = (os.getenv("MONGODB_DB") or "").strip()
    if fallback:
        names.append(fallback)
    if "test" not in [n.lower() for n in names]:
        names.append("test")
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        k = n.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(n)
    return out


def _collections() -> dict[str, list[str]]:
    return {
        "users": ["_id", "email"],
        "bookings": ["_id", "user", "partner", "spaceId"],
        "payments": ["_id", "user", "space"],
        "invoices": ["_id", "user", "partner"],
        "credit_ledger": ["_id", "user"],
        "seatbookings": ["_id", "user", "space", "paymentId"],
        "reviews": ["_id", "user", "space"],
        "properties": ["_id", "partner"],
        "virtualoffices": ["_id", "partner", "property"],
        "coworkingspaces": ["_id", "partner", "property"],
        "meetingrooms": ["_id", "partner", "property"],
    }


def _safe_value(value: Any):
    try:
        from bson import ObjectId  # type: ignore
    except Exception:
        ObjectId = None  # type: ignore

    if ObjectId is not None and isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [_safe_value(v) for v in value[:20]]
    if isinstance(value, dict):
        # Normalize Mongo extended JSON date/oid wrappers from local dump files.
        if set(value.keys()) == {"$date"}:
            return value.get("$date")
        if set(value.keys()) == {"$oid"}:
            return value.get("$oid")

        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in SENSITIVE_KEYS:
                continue
            out[k] = _safe_value(v)
        return out
    return value


def _object_id_str(value: Any) -> str:
    if isinstance(value, dict):
        oid = value.get("$oid")
        if isinstance(oid, str):
            return oid
    safe = _safe_value(value)
    if isinstance(safe, str):
        return safe
    if isinstance(safe, dict):
        oid = safe.get("$oid")
        if isinstance(oid, str):
            return oid
    return ""


@lru_cache(maxsize=1)
def _load_local_dump() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    base = (os.getenv("RELATIONSHIP_JSON_DIR") or "").strip()
    if not base:
        return out
    p = Path(base)
    if not p.exists() or not p.is_dir():
        return out
    for fp in p.glob("test.*.json"):
        name = fp.name[len("test.") : -len(".json")]
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, list):
            out[name] = [d for d in raw if isinstance(d, dict)]
    return out


def _local_find_user(*, emails: list[str], oids: list[str]) -> dict[str, Any] | None:
    data = _load_local_dump()
    users = data.get("users") or []
    if not users:
        return None

    email_set = {e.lower() for e in emails}
    oid_set = {o.lower() for o in oids}
    for u in users:
        if not isinstance(u, dict):
            continue
        user_email = str(u.get("email") or "").strip().lower()
        user_oid = _object_id_str(u.get("_id")).lower()
        if (user_email and user_email in email_set) or (user_oid and user_oid in oid_set):
            return u
    return None


def _local_count_by_user(collection: str, user_id: str) -> int:
    data = _load_local_dump()
    docs = data.get(collection) or []
    total = 0
    target = (user_id or "").lower()
    for d in docs:
        if not isinstance(d, dict):
            continue
        ref = d.get("user")
        ref_id = _object_id_str(ref).lower()
        if ref_id == target:
            total += 1
    return total


def _is_virtual_space_query(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    return bool(VIRTUAL_SPACE_INTENT_RE.search(q))


def _is_detail_intent_query(query: str) -> bool:
    return bool(DETAIL_INTENT_RE.search((query or "").strip()))


def _summarize_doc(coll: str, doc: dict[str, Any]) -> dict[str, Any]:
    base = _safe_value(doc)
    if not isinstance(base, dict):
        return {}

    selected: list[str]
    if coll == "users":
        selected = ["_id", "email", "fullName", "role", "isActive", "kycVerified", "credits", "createdAt"]
    elif coll == "bookings":
        selected = ["_id", "bookingNumber", "type", "status", "user", "partner", "spaceId", "startDate", "endDate"]
    elif coll == "payments":
        selected = ["_id", "status", "paymentType", "spaceModel", "user", "userEmail", "amount", "currency", "spaceName", "createdAt"]
    elif coll == "invoices":
        selected = ["_id", "invoiceNumber", "status", "total", "user", "partner", "createdAt"]
    elif coll == "credit_ledger":
        selected = ["_id", "user", "amount", "type", "description", "balanceAfter", "isExpired", "createdAt"]
    elif coll == "seatbookings":
        selected = ["_id", "user", "space", "status", "paymentId", "startTime", "endTime", "totalAmount"]
    elif coll == "reviews":
        selected = ["_id", "user", "space", "spaceModel", "rating", "comment", "createdAt"]
    else:
        selected = ["_id", "name", "status", "partner", "property", "createdAt", "updatedAt"]

    return {k: base.get(k) for k in selected if k in base}


def build_virtual_business_spaces_context(query: str, *, max_rows: int = 8) -> str:
    if not _is_virtual_space_query(query):
        return ""

    client = _get_client()
    if client is None:
        return ""

    try:
        from bson import ObjectId  # type: ignore
    except Exception:
        ObjectId = None  # type: ignore

    lines: list[str] = []
    total_rows = 0

    for db_name in _db_candidates():
        db = client[db_name]
        try:
            virtual_cursor = (
                db["virtualoffices"]
                .find(
                    {
                        "isDeleted": {"$ne": True},
                        "isActive": {"$ne": False},
                        "approvalStatus": {"$in": ["active", "completed"]},
                    }
                )
                .limit(max_rows)
            )
            virtual_docs = list(virtual_cursor)
        except Exception:
            continue

        if not virtual_docs:
            continue

        property_ids: list[Any] = []
        for d in virtual_docs:
            p = d.get("property")
            if p is not None:
                property_ids.append(p)

        properties_map: dict[str, dict[str, Any]] = {}
        if property_ids and ObjectId is not None:
            try:
                props = list(
                    db["properties"].find(
                        {
                            "_id": {"$in": property_ids},
                            "isDeleted": {"$ne": True},
                            "isActive": {"$ne": False},
                        }
                    )
                )
                for p in props:
                    properties_map[_object_id_str(p.get("_id"))] = _safe_value(p)
            except Exception:
                properties_map = {}

        rows: list[dict[str, Any]] = []
        for v in virtual_docs:
            sv = _safe_value(v)
            if not isinstance(sv, dict):
                continue

            property_id = _object_id_str(sv.get("property"))
            prop = properties_map.get(property_id, {})

            row = {
                "spaceId": _object_id_str(sv.get("_id")),
                "name": prop.get("name"),
                "city": prop.get("city"),
                "area": prop.get("area"),
                "address": prop.get("address"),
                "approvalStatus": sv.get("approvalStatus"),
                "finalGstPricePerYear": sv.get("finalGstPricePerYear"),
                "finalMailingPricePerYear": sv.get("finalMailingPricePerYear"),
                "finalBrPricePerYear": sv.get("finalBrPricePerYear"),
                "partnerId": _object_id_str(sv.get("partner")),
            }
            rows.append(row)

        if rows:
            if not lines:
                lines.append("Business virtual spaces (MongoDB):")
            lines.append(f"- Database `{db_name}`:")
            for row in rows[:max_rows]:
                lines.append(f"  - {row}")
            total_rows += len(rows[:max_rows])

    if not lines:
        return "Business virtual spaces (MongoDB):\n- No active business virtual office spaces found."

    lines.append(f"- Total virtual spaces shown: {total_rows}")
    return "\n".join(lines)


def build_user_identity_response(query: str) -> str:
    emails, oids = _extract_terms(query)
    if not emails and not oids:
        return ""

    client = _get_client()
    if client is None:
        return ""

    try:
        from bson import ObjectId  # type: ignore
    except Exception:
        ObjectId = None  # type: ignore

    for db_name in _db_candidates():
        db = client[db_name]
        user_doc: dict[str, Any] | None = None

        if emails:
            try:
                user_doc = db["users"].find_one({"email": {"$in": emails}})
            except Exception:
                user_doc = None

        if user_doc is None and oids and ObjectId is not None:
            for oid in oids:
                try:
                    user_doc = db["users"].find_one({"_id": ObjectId(oid)})
                except Exception:
                    user_doc = None
                if user_doc is not None:
                    break

        if user_doc is None:
            continue

        user = _safe_value(user_doc)
        if not isinstance(user, dict):
            continue

        user_id = _object_id_str(user.get("_id"))
        full_name = str(user.get("fullName") or "User")
        email = str(user.get("email") or "")
        role = str(user.get("role") or "user")
        is_active = bool(user.get("isActive", False))
        kyc_verified = bool(user.get("kycVerified", False))
        credits = user.get("credits")
        created_at = user.get("createdAt")

        counts: dict[str, int] = {}
        try:
            if ObjectId is not None and user_id:
                oid_value = ObjectId(user_id)
                counts["bookings"] = db["bookings"].count_documents({"user": oid_value})
                counts["payments"] = db["payments"].count_documents({"user": oid_value})
                counts["invoices"] = db["invoices"].count_documents({"user": oid_value})
                counts["reviews"] = db["reviews"].count_documents({"user": oid_value})
        except Exception:
            counts = {}

        lines: list[str] = []
        lines.append(f"Welcome, {full_name}!")
        lines.append("Here are your account details:")
        lines.append(f"- Name: {full_name}")
        if email:
            lines.append(f"- Email: {email}")
        if user_id:
            lines.append(f"- User ID: {user_id}")
        lines.append(f"- Role: {role}")
        lines.append(f"- Account status: {'Active' if is_active else 'Inactive'}")
        lines.append(f"- KYC status: {'Verified' if kyc_verified else 'Not verified'}")
        if credits is not None:
            lines.append(f"- Credits: {credits}")
        if created_at:
            lines.append(f"- Joined on: {created_at}")
        if counts:
            lines.append("- Usage summary:")
            lines.append(f"  - Bookings: {counts.get('bookings', 0)}")
            lines.append(f"  - Payments: {counts.get('payments', 0)}")
            lines.append(f"  - Invoices: {counts.get('invoices', 0)}")
            lines.append(f"  - Reviews: {counts.get('reviews', 0)}")
        lines.append("Would you like your latest bookings and payment history too?")
        return "\n".join(lines)

    return ""


def process_user_identity_verification(query: str, *, session_id: str, role: str = "user") -> str:
    query = (query or "").strip()
    if not query:
        return ""
    role_norm = (role or "").strip().lower()
    gated_roles = {"admin", "affiliate", "user", "partner"}

    # Guest role should bypass identity/password gate and continue normal chat.
    if role_norm == "guest":
        _PENDING_IDENTITY.pop(session_id, None)
        return ""

    # Step 2: password verification if session already in challenge state.
    pending = _PENDING_IDENTITY.get(session_id)
    if pending:
        password_input = _extract_password_input(query)
        if not password_input:
            return "Please enter your password to continue verification."
        password_input = _normalize_bcrypt_like(password_input)

        db_name = pending.get("db_name", "")
        user_id = pending.get("user_id", "")
        if not db_name or not user_id:
            _PENDING_IDENTITY.pop(session_id, None)
            return "Verification session expired. Please share your email or user ID again."

        client = _get_client()
        user_doc: dict[str, Any] | None = None
        ObjectId = None  # type: ignore
        if db_name == "local_json":
            user_doc = _local_find_user(emails=[], oids=[user_id])
        else:
            if client is not None:
                try:
                    from bson import ObjectId as _ObjectId  # type: ignore

                    ObjectId = _ObjectId  # type: ignore
                    user_doc = client[db_name]["users"].find_one({"_id": ObjectId(user_id)})
                except Exception:
                    user_doc = None

            if user_doc is None:
                # Fallback to local JSON dump when Mongo is unreachable.
                user_doc = _local_find_user(emails=[], oids=[user_id])

        if not user_doc:
            _PENDING_IDENTITY.pop(session_id, None)
            return "User not found anymore. Please share your email or user ID again."

        db_hash = str(user_doc.get("password") or "")
        verified = False

        # Guardrail for obvious fragmented input.
        if len(password_input) < 8:
            return "Please enter the full password in one message (example: password: your_password)."

        # Testing convenience: allow exact bcrypt hash pasted from DB.
        if db_hash and password_input == db_hash:
            verified = True

        if db_hash.startswith("$2"):
            # Accept normalized bcrypt-like text too.
            if not verified:
                normalized_hash_input = _normalize_bcrypt_like(password_input)
                if normalized_hash_input == db_hash:
                    verified = True
            if not verified:
                try:
                    import bcrypt  # type: ignore

                    verified = bcrypt.checkpw(password_input.encode("utf-8"), db_hash.encode("utf-8"))
                except Exception:
                    verified = False
        else:
            verified = password_input == db_hash

        if not verified:
            return "Password verification failed. Please enter your original account password in one line."

        _PENDING_IDENTITY.pop(session_id, None)
        user = _safe_value(user_doc)
        if not isinstance(user, dict):
            return "Verified. Unable to read profile details right now."

        full_name = str(user.get("fullName") or "User")
        email = str(user.get("email") or "")
        role = str(user.get("role") or "user")
        is_active = bool(user.get("isActive", False))
        kyc_verified = bool(user.get("kycVerified", False))
        credits = user.get("credits")
        created_at = user.get("createdAt")

        counts = {"bookings": 0, "payments": 0, "invoices": 0, "reviews": 0}
        if client is not None and ObjectId is not None:
            try:
                oid_value = ObjectId(user_id)
                counts = {
                    "bookings": client[db_name]["bookings"].count_documents({"user": oid_value}),
                    "payments": client[db_name]["payments"].count_documents({"user": oid_value}),
                    "invoices": client[db_name]["invoices"].count_documents({"user": oid_value}),
                    "reviews": client[db_name]["reviews"].count_documents({"user": oid_value}),
                }
            except Exception:
                counts = {"bookings": 0, "payments": 0, "invoices": 0, "reviews": 0}
        else:
            counts = {
                "bookings": _local_count_by_user("bookings", user_id),
                "payments": _local_count_by_user("payments", user_id),
                "invoices": _local_count_by_user("invoices", user_id),
                "reviews": _local_count_by_user("reviews", user_id),
            }

        lines: list[str] = []
        lines.append(f"Welcome, {full_name}!")
        lines.append("Your verified account details:")
        lines.append(f"- Name: {full_name}")
        if email:
            lines.append(f"- Email: {email}")
        lines.append(f"- User ID: {user_id}")
        lines.append(f"- Role: {role}")
        lines.append(f"- Account status: {'Active' if is_active else 'Inactive'}")
        lines.append(f"- KYC status: {'Verified' if kyc_verified else 'Not verified'}")
        if credits is not None:
            lines.append(f"- Credits: {credits}")
        if created_at:
            lines.append(f"- Joined on: {created_at}")
        lines.append("- Usage summary:")
        lines.append(f"  - Bookings: {counts.get('bookings', 0)}")
        lines.append(f"  - Payments: {counts.get('payments', 0)}")
        lines.append(f"  - Invoices: {counts.get('invoices', 0)}")
        lines.append(f"  - Reviews: {counts.get('reviews', 0)}")
        lines.append("")
        lines.append("How may I assist you today?")
        return "\n".join(lines)

    # Step 1: ask for identifier if user requested details but did not provide email/id.
    emails, oids = _extract_terms(query)
    if role_norm not in gated_roles:
        return ""

    if _is_detail_intent_query(query) and not emails and not oids:
        return "Please share your email ID or 24-character user ID to continue."

    # Step 1b: if identifier provided, start challenge.
    if not emails and not oids:
        return ""

    # Fast path: local JSON dump first (avoids DNS wait if Mongo is down).
    local_user = _local_find_user(emails=emails, oids=oids)
    if local_user:
        user = _safe_value(local_user)
        if isinstance(user, dict):
            uid = _object_id_str(user.get("_id"))
            if uid:
                _PENDING_IDENTITY[session_id] = {"db_name": "local_json", "user_id": uid}
                return "Account found. Please enter your password to verify identity."

    client = _get_client()
    ObjectId = None  # type: ignore
    if client is not None:
        try:
            from bson import ObjectId as _ObjectId  # type: ignore

            ObjectId = _ObjectId  # type: ignore
        except Exception:
            ObjectId = None  # type: ignore

    for db_name in _db_candidates():
        user_doc: dict[str, Any] | None = None
        if client is None:
            break
        db = client[db_name]

        if emails:
            try:
                user_doc = db["users"].find_one({"email": {"$in": emails}})
            except Exception:
                user_doc = None

        if user_doc is None and oids and ObjectId is not None:
            for oid in oids:
                try:
                    user_doc = db["users"].find_one({"_id": ObjectId(oid)})
                except Exception:
                    user_doc = None
                if user_doc is not None:
                    break

        if user_doc is None:
            continue

        user = _safe_value(user_doc)
        if not isinstance(user, dict):
            continue
        uid = _object_id_str(user.get("_id"))
        if not uid:
            continue

        _PENDING_IDENTITY[session_id] = {"db_name": db_name, "user_id": uid}
        return "Account found. Please enter your password to verify identity."

    return "No account found for that email/user ID. Please check and try again."


def build_live_lookup_context(query: str, *, max_rows_per_collection: int = 3) -> str:
    emails, oids = _extract_terms(query)
    if not emails and not oids:
        return ""

    client = _get_client()
    if client is None:
        return ""

    try:
        from bson import ObjectId  # type: ignore
    except Exception:
        ObjectId = None  # type: ignore

    collection_fields = _collections()
    lines: list[str] = []
    lines.append("Live lookup results (MongoDB):")
    lines.append(f"- Query entities: emails={emails or []}, ids={oids or []}")

    total_hits = 0
    for db_name in _db_candidates():
        db = client[db_name]
        db_lines: list[str] = []

        for coll, fields in collection_fields.items():
            query_or: list[dict[str, Any]] = []

            if emails and coll == "users":
                query_or.append({"email": {"$in": emails}})
            if emails and coll == "payments":
                query_or.append({"userEmail": {"$in": emails}})

            for oid in oids:
                if "_id" in fields and ObjectId is not None:
                    try:
                        query_or.append({"_id": ObjectId(oid)})
                    except Exception:
                        pass
                for f in fields:
                    if f == "_id":
                        continue
                    if f == "paymentId":
                        query_or.append({f: oid})
                    elif ObjectId is not None:
                        try:
                            query_or.append({f: ObjectId(oid)})
                        except Exception:
                            pass

            if not query_or:
                continue

            try:
                cursor = db[coll].find({"$or": query_or}).limit(max_rows_per_collection)
                docs = list(cursor)
            except Exception:
                continue
            if not docs:
                continue

            total_hits += len(docs)
            summarized = [_summarize_doc(coll, d) for d in docs]
            db_lines.append(f"  - {coll}: {summarized}")

        if db_lines:
            lines.append(f"- Database `{db_name}`:")
            lines.extend(db_lines)

    if total_hits == 0:
        lines.append("- No matching records found in configured lookup collections.")
    else:
        lines.append(f"- Total matched docs: {total_hits}")

    return "\n".join(lines)
