import contextvars
from typing import Any, Dict, Optional

from knowledgator_mcp.auth.base import BaseAuthProvider

_current_request_key: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_request_key", default=None
)


def set_current_request_key(key: Optional[str]) -> contextvars.Token:
    """Store the extracted API key strictly for the active request lifecycle."""
    clean_key = key.strip() if key and isinstance(key, str) else None
    return _current_request_key.set(clean_key)


def reset_current_request_key(token: contextvars.Token) -> None:
    """Reset the request-scoped API key immediately when request ends."""
    try:
        _current_request_key.reset(token)
    except Exception:
        _current_request_key.set(None)


def get_current_request_key() -> Optional[str]:
    """Retrieve the API key associated strictly with the active request task."""
    return _current_request_key.get()


def extract_key_from_ctx(ctx: Any) -> Optional[str]:
    """Extract API key from FastMCP context, native Starlette request."""
    # Check HTTP request header captured by per-request ASGI middleware
    captured_key = get_current_request_key()
    if captured_key:
        return captured_key

    if ctx is None:
        return None

    # Handle dictionary context
    if isinstance(ctx, dict):
        headers = ctx.get("headers", {})
        meta = ctx.get("meta", {})
        key = (
            headers.get("KnowledgatorAPI-Key")
            or headers.get("knowledgatorapi-key")
            or headers.get("authorization")
            or meta.get("api_key")
        )
        if isinstance(key, str) and key.strip():
            if key.startswith("Bearer "):
                return key[7:].strip()
            return key.strip()
        return None

    # Handle FastMCP Context object
    req_ctx = getattr(ctx, "request_context", None)
    candidate_headers = []
    if req_ctx is not None:
        raw_req = getattr(req_ctx, "request", None)
        if raw_req is not None and hasattr(raw_req, "headers"):
            candidate_headers.append(raw_req.headers)
        if hasattr(req_ctx, "headers"):
            candidate_headers.append(req_ctx.headers)
    if hasattr(ctx, "headers"):
        candidate_headers.append(ctx.headers)

    meta = getattr(req_ctx, "meta", None) if req_ctx is not None else getattr(ctx, "meta", None)

    key = None
    for headers in candidate_headers:
        if headers and hasattr(headers, "get"):
            val = (
                headers.get("KnowledgatorAPI-Key")
                or headers.get("knowledgatorapi-key")
                or headers.get("authorization")
            )
            if isinstance(val, str) and val.strip():
                key = val
                break

    if not key and meta and hasattr(meta, "get"):
        val = meta.get("api_key")
        if isinstance(val, str) and val.strip():
            key = val

    if isinstance(key, str) and key.strip():
        if key.startswith("Bearer "):
            return key[7:].strip()
        return key.strip()

    return None


class APIKeyAuthProvider(BaseAuthProvider):
    """Authentication provider using Knowledgator API key."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.knowledgator.com"):
        self.api_key = api_key
        self.base_url = base_url

    def get_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["KnowledgatorAPI-Key"] = self.api_key
        return headers

    def get_base_url(self) -> str:
        return self.base_url


class APIProxyAuthProvider(BaseAuthProvider):
    """Stateless reverse proxy auth provider extracting API keys per-request from FastMCP Context."""

    def __init__(
        self,
        ctx: Any = None,
        base_url: str = "https://api.knowledgator.com",
    ):
        self.ctx = ctx
        self.base_url = base_url

    def resolve_api_key(self) -> Optional[str]:
        # Extract per-request key strictly from FastMCP Context
        return extract_key_from_ctx(self.ctx)

    def get_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        key = self.resolve_api_key()
        if key:
            headers["KnowledgatorAPI-Key"] = key
        return headers

    def get_base_url(self) -> str:
        return self.base_url


