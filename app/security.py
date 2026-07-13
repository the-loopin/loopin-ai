"""Internal API authentication and request correlation helpers."""

from __future__ import annotations

import os
import secrets
from contextvars import ContextVar
from dataclasses import dataclass

from fastapi import HTTPException, Request


_request_id: ContextVar[str] = ContextVar("request_id", default="-")


class SecurityConfigError(ValueError):
    """Raised when a deployment has no inference authentication configured."""


@dataclass(frozen=True)
class SecuritySettings:
    service_token: str
    production: bool

    @classmethod
    def from_environment(cls) -> "SecuritySettings":
        token = os.getenv("LOOPIN_SERVICE_TOKEN", "")
        if not token.strip():
            raise SecurityConfigError(
                "LOOPIN_SERVICE_TOKEN must be configured for internal API authentication."
            )
        environment = os.getenv("LOOPIN_ENV", "development").strip().lower()
        return cls(service_token=token, production=environment == "production")


def request_id() -> str:
    return _request_id.get()


def set_request_id(value: str):
    return _request_id.set(value)


def reset_request_id(token) -> None:
    _request_id.reset(token)


def inbound_request_id(value: str | None) -> str:
    """Accept only a bounded printable correlation ID supplied by a caller."""
    normalized = value.replace("-", "").replace("_", "").replace(".", "") if value else ""
    if value and len(value) <= 128 and value.isascii() and normalized.isalnum():
        return value
    return secrets.token_hex(16)


def require_service_token(request: Request) -> None:
    """Authorize a trusted service-to-service caller without logging credentials."""
    configured = request.app.state.security.service_token
    authorization = request.headers.get("authorization", "")
    bearer = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    supplied = request.headers.get("x-loopin-service-token", bearer)
    if not supplied or not secrets.compare_digest(supplied, configured):
        raise HTTPException(status_code=401, detail="Invalid service authentication.")
