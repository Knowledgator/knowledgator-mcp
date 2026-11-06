from typing import Any, Dict, Optional

from knowledgator_mcp.auth.base import BaseAuthProvider


def extract_key_from_ctx(ctx: Any) -> Optional[str]:
    """Extract API key from FastMCP context."""
    if ctx is None:
        return None

    # FastMCP Context -> request_context.request is Starlette Request
    req_ctx = getattr(ctx, "request_context", None)
    if req_ctx:
        request = getattr(req_ctx, "request", None)
        if request and hasattr(request, "headers"):
            key = request.headers.get("knowledgatorapi-key")
            if key:
                return key

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
    """Stateless auth provider extracting API keys per-request from FastMCP Context."""

    def __init__(
        self,
        ctx: Any = None,
        base_url: str = "https://api.knowledgator.com",
    ):
        self.ctx = ctx
        self.base_url = base_url

    def resolve_api_key(self) -> Optional[str]:
        return extract_key_from_ctx(self.ctx)

    def get_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        key = self.resolve_api_key()
        if key:
            headers["KnowledgatorAPI-Key"] = key
        return headers

    def get_base_url(self) -> str:
        return self.base_url
