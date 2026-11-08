"""
Unit tests verifying DNS rebinding protection and transport security configuration.
All tests run 100% offline with zero API key dependencies.
"""
import pytest
from starlette.requests import Request
from starlette.responses import Response
from starlette.datastructures import Headers
from mcp.server.transport_security import (
    TransportSecuritySettings,
    TransportSecurityMiddleware,
)

from knowledgator_mcp.config import ServerConfig


def _build_mock_request(host: str = "localhost:8000", origin: str | None = None, method: str = "GET") -> Request:
    """Helper to build a Starlette Request with specific headers for unit testing."""
    headers = [("host", host)]
    if origin:
        headers.append(("origin", origin))
    if method == "POST":
        headers.append(("content-type", "application/json"))

    scope = {
        "type": "http",
        "method": method,
        "path": "/sse",
        "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers],
    }
    return Request(scope)


def test_transport_security_config_defaults(monkeypatch):
    """Test default ServerConfig transport security settings."""
    monkeypatch.delenv("KG_MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("KG_MCP_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("KG_MCP_ENABLE_DNS_REBINDING_PROTECTION", raising=False)

    config = ServerConfig(_env_file=None)
    assert config.enable_dns_rebinding_protection is True
    assert "localhost" in config.allowed_hosts
    assert "127.0.0.1:*" in config.allowed_hosts
    assert config.allowed_origins == []


def test_transport_security_config_comma_separated_parsing():
    """Test parsing comma-separated strings from environment."""
    config = ServerConfig(
        _env_file=None,
        allowed_hosts="mcp.example.com, knowledgator-mcp:8000, knowledgator-mcp.default.svc.cluster.local:*",
        allowed_origins="https://example.ai, https://mcp.example.com",
    )
    assert config.allowed_hosts == [
        "mcp.example.com",
        "knowledgator-mcp:8000",
        "knowledgator-mcp.default.svc.cluster.local:*",
    ]
    assert config.allowed_origins == [
        "https://example.ai",
        "https://mcp.example.com",
    ]


def test_transport_security_config_json_array_parsing():
    """Test parsing JSON array strings."""
    config = ServerConfig(
        _env_file=None,
        allowed_hosts='["mcp.example.com", "127.0.0.1:*"]',
    )
    assert config.allowed_hosts == ["mcp.example.com", "127.0.0.1:*"]


@pytest.mark.asyncio
async def test_middleware_allows_configured_host():
    """Allowed host passes validation with None response."""
    settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["mcp.example.com", "mcp.example.com:*", "localhost:*"],
    )
    middleware = TransportSecurityMiddleware(settings)

    # Exact match
    req1 = _build_mock_request(host="mcp.example.com")
    res1 = await middleware.validate_request(req1)
    assert res1 is None

    # Wildcard port match
    req2 = _build_mock_request(host="mcp.example.com:443")
    res2 = await middleware.validate_request(req2)
    assert res2 is None


@pytest.mark.asyncio
async def test_middleware_blocks_unauthorized_host():
    """Untrusted host fails validation returning 421 Misdirected Request."""
    settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["mcp.example.com"],
    )
    middleware = TransportSecurityMiddleware(settings)

    # Untrusted domain attempting DNS rebinding
    req = _build_mock_request(host="evil-attacker.com")
    res = await middleware.validate_request(req)

    assert isinstance(res, Response)
    assert res.status_code == 421


@pytest.mark.asyncio
async def test_middleware_k8s_internal_hosts():
    """Internal cluster service names pass when configured."""
    settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "mcp.example.com:*",
            "knowledgator-mcp.mcp-system.svc.cluster.local:*",
            "knowledgator-mcp:8000",
        ],
    )
    middleware = TransportSecurityMiddleware(settings)

    # Sandbox calling internal service
    req = _build_mock_request(host="knowledgator-mcp.mcp-system.svc.cluster.local:8000")
    res = await middleware.validate_request(req)
    assert res is None


@pytest.mark.asyncio
async def test_middleware_disabled_protection_allows_all():
    """When protection is explicitly disabled, all hosts pass."""
    settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=[],
    )
    middleware = TransportSecurityMiddleware(settings)

    req = _build_mock_request(host="random-untrusted-host.com")
    res = await middleware.validate_request(req)
    assert res is None


@pytest.mark.asyncio
async def test_origin_validation():
    """Origins are validated against allowed list when present."""
    settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["mcp.example.com"],
        allowed_origins=["https://example.ai"],
    )
    middleware = TransportSecurityMiddleware(settings)

    # Allowed origin passes
    req_allowed = _build_mock_request(host="mcp.example.com", origin="https://example.ai")
    res_allowed = await middleware.validate_request(req_allowed)
    assert res_allowed is None

    # Untrusted origin gets rejected with 400
    req_blocked = _build_mock_request(host="mcp.example.com", origin="https://evil-site.com")
    res_blocked = await middleware.validate_request(req_blocked)
    assert isinstance(res_blocked, Response)
    assert res_blocked.status_code == 400
