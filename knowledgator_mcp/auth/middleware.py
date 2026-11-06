"""ASGI middleware for capturing per-request API keys in proxy mode."""

from typing import Any
from knowledgator_mcp.auth.providers import reset_current_request_key, set_current_request_key


class APIKeyCaptureMiddleware:
    """Pure ASGI middleware to capture Knowledgator API keys strictly per-request.

    Ensures zero cross-tenant leakage: keys are isolated to the asyncio ContextVar
    and immediately reset when the HTTP request finishes.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        token = None
        if scope.get("type") == "http":
            for name, value in scope.get("headers", []):
                lower_name = name.lower()
                if lower_name in (b"knowledgatorapi-key", b"authorization"):
                    try:
                        key_str = value.decode("utf-8").strip()
                        if key_str.startswith("Bearer "):
                            key_str = key_str[7:].strip()
                        if key_str:
                            token = set_current_request_key(key_str)
                    except Exception:
                        pass
                    break
        try:
            await self.app(scope, receive, send)
        finally:
            if token is not None:
                reset_current_request_key(token)
