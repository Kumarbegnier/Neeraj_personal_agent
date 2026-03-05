import os
import uuid
import logging
from typing import Optional, List

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, Session

from app.models import Base, ChatSession, ChatMessage

logger = logging.getLogger(__name__)


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./chat.db"
)

_engine_kwargs = {"pool_pre_ping": True}
if (DATABASE_URL or "").startswith("sqlite"):
    # Required for FastAPI/uvicorn where sync endpoints run in a threadpool.
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)

try:
    url = make_url(DATABASE_URL)
    if url.drivername.startswith("sqlite") and url.database:
        db_path = url.database
        if not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)
        logger.info("SQLite DB path: %s", db_path)
except Exception:
    logger.exception("Failed to parse DATABASE_URL")

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base.metadata.create_all(bind=engine)


def _get_db() -> Session:
    return SessionLocal()


# --------------------------------------------------
# Create Session Per User
# --------------------------------------------------
def create_session(user_id: str, tenant_id: str) -> str:

    db: Optional[Session] = None

    try:
        db = _get_db()

        session = ChatSession(
            user_id=user_id,
            tenant_id=tenant_id
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        return session.id

    except Exception:
        logger.exception("DB create_session failed")
        return str(uuid.uuid4())

    finally:
        if db:
            db.close()


def ensure_session(session_id: str, user_id: str, tenant_id: str) -> None:
    """Ensure a chat session row exists for the given session_id.

    This is useful when session IDs are derived deterministically (e.g. from
    tenant/user/conversation_id) and we don't want to create a new UUID session
    on every request.
    """

    if not (session_id or "").strip():
        return

    db: Optional[Session] = None
    try:
        db = _get_db()

        existing = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id)
            .first()
        )
        if existing is not None:
            return

        session = ChatSession(
            id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        db.add(session)
        db.commit()

    except Exception:
        logger.exception("DB ensure_session failed")

    finally:
        if db:
            db.close()


# --------------------------------------------------
# Save Chat Message
# --------------------------------------------------
def save_message(session_id: str, role: str, content: str):

    db: Optional[Session] = None

    try:
        db = _get_db()

        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content
        )

        db.add(msg)
        db.commit()

    except Exception:
        logger.exception("DB save_message failed")

    finally:
        if db:
            db.close()


# --------------------------------------------------
# Get Session Messages (History)
# --------------------------------------------------
def get_session_messages(session_id: str) -> List[ChatMessage]:

    db: Optional[Session] = None

    try:
        db = _get_db()

        msgs = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

        return msgs

    except Exception:
        logger.exception("DB get_session_messages failed")
        return []

    finally:
        if db:
            db.close()


def clear_session_messages(session_id: str) -> None:
    """Delete all messages for a session."""
    db: Optional[Session] = None
    try:
        db = _get_db()
        (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .delete(synchronize_session=False)
        )
        db.commit()
    except Exception:
        logger.exception("DB clear_session_messages failed")
    finally:
        if db:
            db.close()