import os
import logging
import re
from typing import Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app.chains import build_chain
from app.session_store import ensure_session, get_session_messages

from app.auth import AuthContext, get_auth_context
from app.vectorstore import format_docs, retrieve_with_scores
from app.relationships import get_relationship_context
from app.live_lookup import (
    build_live_lookup_context,
    build_virtual_business_spaces_context,
    process_user_identity_verification,
)
from app.booking_flow import process_booking_discovery_flow


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

ROLE_CONTEXT_RE = re.compile(r"^\s*\[role_context=([a-zA-Z_]+)\]\s*")


app = FastAPI()


chain = build_chain()


class Query(BaseModel):
    query: str
    conversation_id: str = Field(default="default", min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID"
    )

@app.post("/chat")
def chat(
    data: Query,
    auth: AuthContext = Depends(get_auth_context),
):
    try:

    
        session_id = (data.session_id or "").strip()
        if not session_id:
            
            session_id = f"{auth.tenant_id}:{auth.user_id}:{data.conversation_id}"

        # for backward compatibility, ensure session exists even if session_id is not provided (using conversation_id as part of session_id)
        ensure_session(
            session_id=session_id,
            user_id=auth.user_id,
            tenant_id=auth.tenant_id,
        )



        raw_query = (data.query or "").strip()
        role_context = ""
        m = ROLE_CONTEXT_RE.match(raw_query)
        if m:
            role_context = m.group(1).strip().lower()
            raw_query = ROLE_CONTEXT_RE.sub("", raw_query, count=1).strip()

        effective_role = role_context or auth.role
        namespace = auth.namespace
        # # session_id = f"{auth.tenant_id}:{auth.user_id}:{data.conversation_id}"
        # session_id = auth.session_id

        logger.info("/chat namespace=%s session_id=%s role=%s query=%r", namespace, session_id, effective_role, raw_query)

        try:
            identity_response = process_user_identity_verification(
                raw_query,
                session_id=session_id,
                role=effective_role,
            )
            if identity_response:
                logger.info("identity_lookup_hit session_id=%s", session_id)
                return {"reply": identity_response, "session_id": session_id}
        except Exception:
            logger.exception("Identity lookup failed; continuing with normal flow")

        try:
            booking_flow_response = process_booking_discovery_flow(raw_query, session_id=session_id)
            if booking_flow_response:
                logger.info("booking_flow_hit session_id=%s", session_id)
                return {"reply": booking_flow_response, "session_id": session_id}
        except Exception:
            logger.exception("Booking flow failed; continuing with normal flow")

    

        docs = []
        context = ""
        relationship_context = ""
        live_lookup_context = ""
        virtual_spaces_context = ""

        try:
            docs, _scores = retrieve_with_scores(namespace=namespace, query=raw_query, k=4)
            context = format_docs(docs)
            logger.info("retrieved_docs=%s context_chars=%s", len(docs), len(context))
        except Exception as exc:
            logger.exception("Retrieval failed; continuing with empty context")
            context = ""

        try:
            relationship_context = get_relationship_context()
        except Exception:
            logger.exception("Relationship context build failed; continuing with empty relationship context")
            relationship_context = ""

        try:
            live_lookup_context = build_live_lookup_context(raw_query)
            if live_lookup_context:
                logger.info("live_lookup_context_chars=%s", len(live_lookup_context))
        except Exception:
            logger.exception("Live lookup failed; continuing with empty live lookup context")
            live_lookup_context = ""

        try:
            virtual_spaces_context = build_virtual_business_spaces_context(raw_query)
            if virtual_spaces_context:
                logger.info("virtual_spaces_context_chars=%s", len(virtual_spaces_context))
        except Exception:
            logger.exception("Virtual spaces lookup failed; continuing with empty virtual spaces context")
            virtual_spaces_context = ""

        response = chain.invoke(
            {
                "question": raw_query,
                "context": context,
                "relationship_context": relationship_context,
                "live_lookup_context": live_lookup_context,
                "virtual_spaces_context": virtual_spaces_context,
                "role_context": effective_role,
            },
            config={"configurable": {"session_id": session_id, "namespace": namespace}}
        )

        payload = {"reply": response.content, "session_id": session_id}

        return payload
    
    except Exception as e:
        logger.exception("Chat failed")
        return {"error": f"{type(e).__name__}: {str(e)}"}
    

@app.get("/health")
def health():
    return {"status": "ok"}
