from typing import Any, Dict, Optional, Tuple

from knowledgator import Knowledgator
from knowledgator.exceptions import ClientError

from knowledgator_mcp.auth import BaseAuthProvider, get_auth_provider
from knowledgator_mcp.config import ServerConfig, settings

from knowledgator_mcp.auth.providers import extract_key_from_ctx

_client: Optional[Knowledgator] = None


def get_kg_client(
    server_config: Optional[ServerConfig] = None,
    auth_provider: Optional[BaseAuthProvider] = None,
    ctx: Any = None,
    force_refresh: bool = False,
) -> Knowledgator:
    """Return an initialized Knowledgator SDK client instance."""
    global _client
    cfg = server_config or settings
    auth = auth_provider or get_auth_provider(cfg, ctx=ctx)
    headers = auth.get_headers()
    base_url = auth.get_base_url()
    api_key = headers.get("KnowledgatorAPI-Key") or cfg.get_effective_api_key()

    if (cfg.auth_type or "").lower() == "api_proxy":
        proxy_api_key = extract_key_from_ctx(ctx)
        if not proxy_api_key:
            raise ClientError(
                "Authentication required. Please provide an API key in request headers (KnowledgatorAPI-Key).",
                status_code=401,
            )
        eff_headers = dict(headers)
        if "KnowledgatorAPI-Key" not in eff_headers:
            eff_headers["KnowledgatorAPI-Key"] = proxy_api_key
        # Create stateless SDK client instance for this request context
        return Knowledgator(
            api_key=proxy_api_key,
            server_url=base_url,
            server_base_headers=eff_headers,
            timeout_ms=cfg.request_timeout_ms,
        )

    if _client is None or force_refresh or server_config is not None or auth_provider is not None:
        client_inst = Knowledgator(
            api_key=api_key,
            server_url=base_url,
            server_base_headers=headers,
            timeout_ms=cfg.request_timeout_ms,
        )
        if server_config is None and auth_provider is None and not force_refresh:
            _client = client_inst
        return client_inst
    return _client


def get_client_context(
    server_config: Optional[ServerConfig] = None,
    auth_provider: Optional[BaseAuthProvider] = None,
    ctx: Any = None,
) -> Tuple[Knowledgator, str, Dict[str, str]]:
    """Return tuple of (SDK client, resolved server_url, headers dict)."""
    cfg = server_config or settings
    auth = auth_provider or get_auth_provider(cfg, ctx=ctx)
    client = get_kg_client(cfg, auth, ctx=ctx)
    return client, auth.get_base_url(), auth.get_headers()


def close_kg_client() -> None:
    """Close the underlying Knowledgator SDK client."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
