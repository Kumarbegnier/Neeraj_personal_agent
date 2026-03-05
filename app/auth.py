import os
from dataclasses import dataclass
from typing import Literal, Optional
import logging

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


Role = Literal["user", "admin", "partner", "affiliate", "sales"]


_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    tenant_id: str
    role: Role


    @property
    def namespace(self) -> str:
        return f"{self.tenant_id}:{self.role}"



def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        logger.error(f"Missing env var: {name}")
        raise HTTPException(
            status_code=500,
            detail="Authentication configuration error"
        )

    return value




def _get_claim(payload: dict, claim_name: str) -> Optional[str]:
    try:
        value = payload.get(claim_name)
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)

    except Exception:
        logger.warning(f"Failed reading claim: {claim_name}")
        return None






def _decode_jwt(token: str) -> dict:

    secret = _required_env("JWT_SECRET").strip()
    algorithms = [os.getenv("JWT_ALGORITHM", "HS256")]
    options = {"verify_signature": True}
    issuer = os.getenv("JWT_ISSUER") or None
    audience = os.getenv("JWT_AUDIENCE") or None

    try:
        return jwt.decode(
            token,
            secret,
            algorithms=algorithms,
            issuer=issuer,
            audience=audience,
            options=options,
        )

    except jwt.ExpiredSignatureError:
        logger.info("JWT expired")
        raise HTTPException(status_code=401, detail="Token expired")

    except jwt.InvalidTokenError:
        logger.info("Invalid JWT")
        raise HTTPException(status_code=401, detail="Invalid token")

    except HTTPException:
        raise

    except Exception:
        logger.exception("JWT decode failed")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )


def get_auth_context(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthContext:
    if os.getenv("DEV_AUTH_BYPASS", "false").lower() == "true":
        logger.warning("DEV AUTH BYPASS ENABLED")
        return AuthContext(
            user_id=os.getenv("DEV_USER_ID", "dev_user"),
            tenant_id=os.getenv("DEV_TENANT_ID", "dev_tenant"),
            role=os.getenv("DEV_ROLE", "admin"),  # type: ignore
        )

    if credentials is None or not (credentials.credentials or "").strip():
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if (credentials.scheme or "").lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization scheme")

    try:
        token = credentials.credentials.strip()
        payload = _decode_jwt(token)

        user_claim = os.getenv("JWT_USER_CLAIM", "sub")
        tenant_claim = os.getenv("JWT_TENANT_CLAIM", "tenant_id")
        role_claim = os.getenv("JWT_ROLE_CLAIM", "role")

        user_id = _get_claim(payload, user_claim)
        tenant_id = _get_claim(payload, tenant_claim)
        role_raw = _get_claim(payload, role_claim)

        if not user_id:
            raise HTTPException(status_code=401, detail=f"Missing claim: {user_claim}")
        if not tenant_id:
            raise HTTPException(status_code=401, detail=f"Missing claim: {tenant_claim}")
        if not role_raw:
            raise HTTPException(status_code=401, detail=f"Missing claim: {role_claim}")

        role_raw = role_raw.lower()
        allowed_roles: set[str] = {"user", "admin", "partner", "affiliate", "sales"}
        if role_raw not in allowed_roles:
            raise HTTPException(status_code=403, detail="Role not allowed")

        return AuthContext(
            user_id=user_id,
            tenant_id=tenant_id,
            role=role_raw,  # type: ignore[arg-type]
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Authentication failed")
        raise HTTPException(status_code=401, detail="Authentication failed")



