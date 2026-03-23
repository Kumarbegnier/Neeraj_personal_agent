from __future__ import annotations

import os
from collections import defaultdict, deque
from threading import RLock
from time import time

from .models import AuthContext, AuthMode, GatewayHeaders, GatewayResult, RateLimitStatus, UserRequest


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class InterfaceGateway:
    def __init__(self) -> None:
        self._window_seconds = int(os.getenv("INTERFACE_RATE_LIMIT_WINDOW_SECONDS", "60"))
        self._request_limit = int(os.getenv("INTERFACE_RATE_LIMIT_PER_WINDOW", "60"))
        self._require_auth = _env_flag("REQUIRE_AUTH", default=False)
        self._dev_auth_bypass = _env_flag("DEV_AUTH_BYPASS", default=False)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = RLock()

    def process(self, request: UserRequest, headers: GatewayHeaders | None = None) -> GatewayResult:
        header_context = headers or GatewayHeaders()
        normalized_message = " ".join(request.message.split())
        auth = self._authenticate(request, header_context)
        rate_limit = self._check_rate_limit(auth.principal_id or request.user_id)
        accepted = rate_limit.allowed and (auth.is_authenticated or not self._require_auth)

        return GatewayResult(
            channel=request.channel,
            client_id=header_context.client_id or f"{request.channel.value}-client",
            accepted=accepted,
            normalized_message=normalized_message,
            auth=auth,
            rate_limit=rate_limit,
            metadata={
                "forwarded_for": header_context.forwarded_for,
                "user_agent": header_context.user_agent,
                "auth_required": self._require_auth,
            },
        )

    def _authenticate(self, request: UserRequest, headers: GatewayHeaders) -> AuthContext:
        if self._dev_auth_bypass:
            return AuthContext(
                mode=AuthMode.bypass,
                is_authenticated=True,
                principal_id=request.user_id or os.getenv("DEV_USER_ID", "dev_user"),
                role=os.getenv("DEV_ROLE", "user"),
                tenant_id=os.getenv("DEV_TENANT_ID", "dev_tenant"),
                reason="DEV_AUTH_BYPASS enabled for local development.",
            )

        if headers.authorization and headers.authorization.lower().startswith("bearer "):
            return AuthContext(
                mode=AuthMode.bearer,
                is_authenticated=True,
                principal_id=request.user_id,
                role="user",
                tenant_id=os.getenv("GUEST_TENANT_ID", "public"),
                reason="Accepted bearer token in interface gateway.",
            )

        if headers.api_key:
            return AuthContext(
                mode=AuthMode.api_key,
                is_authenticated=True,
                principal_id=request.user_id,
                role="user",
                tenant_id=os.getenv("GUEST_TENANT_ID", "public"),
                reason="Accepted API key in interface gateway.",
            )

        return AuthContext(
            mode=AuthMode.anonymous,
            is_authenticated=not self._require_auth,
            principal_id=request.user_id,
            role="guest",
            tenant_id=os.getenv("GUEST_TENANT_ID", "public"),
            reason="No credential provided; proceeding with anonymous scaffold access.",
        )

    def _check_rate_limit(self, principal_id: str) -> RateLimitStatus:
        now = time()
        with self._lock:
            events = self._events[principal_id]
            while events and (now - events[0]) > self._window_seconds:
                events.popleft()

            allowed = len(events) < self._request_limit
            if allowed:
                events.append(now)

            remaining = max(self._request_limit - len(events), 0)
            reason = (
                "Within in-memory gateway budget."
                if allowed
                else "Rate limit exceeded in interface gateway."
            )

            return RateLimitStatus(
                allowed=allowed,
                limit=self._request_limit,
                remaining=remaining,
                window_seconds=self._window_seconds,
                reason=reason,
            )
