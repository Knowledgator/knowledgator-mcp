from typing import Any, Dict, Optional

from knowledgator_mcp.auth.base import BaseAuthProvider


def extract_key_from_ctx(ctx: Any) -> Optional[str]:
    """Extract API key from FastMCP context."""
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
    headers = getattr(req_ctx, "headers", None) if req_ctx is not None else getattr(ctx, "headers", None)
    meta = getattr(req_ctx, "meta", None) if req_ctx is not None else getattr(ctx, "meta", None)

    key = None
    if headers and hasattr(headers, "get"):
        val = (
            headers.get("KnowledgatorAPI-Key")
            or headers.get("knowledgatorapi-key")
            or headers.get("authorization")
        )
        if isinstance(val, str) and val.strip():
            key = val

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


