import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledgator.exceptions import ClientError
from knowledgator_mcp.client import get_kg_client, get_client_context
from knowledgator_mcp.config import ServerConfig
from knowledgator_mcp.auth.factory import get_auth_provider
from knowledgator_mcp.auth.providers import APIProxyAuthProvider
from knowledgator_mcp.tools.inference_tools import instruct_inference


def make_mock_context(headers_dict):
    """Helper to construct a mock FastMCP Context with request_context.headers."""
    mock_req_ctx = MagicMock()
    mock_req_ctx.headers = headers_dict
    mock_ctx = MagicMock()
    mock_ctx.request_context = mock_req_ctx
    return mock_ctx


def test_api_proxy_provider_header_extraction():
    """Verify KnowledgatorAPI-Key header extraction from request context."""
    config = ServerConfig(auth_type="api_proxy", base_url="https://mcp.example.com")
    ctx = make_mock_context({"knowledgatorapi-key": "kg_alice_123"})

    provider = get_auth_provider(config, ctx=ctx)
    assert isinstance(provider, APIProxyAuthProvider)
    headers = provider.get_headers()
    assert headers["KnowledgatorAPI-Key"] == "kg_alice_123"


def test_api_proxy_provider_authorization_bearer_fallback():
    """Verify Authorization: Bearer fallback extraction when KnowledgatorAPI-Key is absent."""
    config = ServerConfig(auth_type="api_proxy", base_url="https://mcp.example.com")
    ctx = make_mock_context({"authorization": "Bearer kg_bearer_token_789"})

    provider = get_auth_provider(config, ctx=ctx)
    headers = provider.get_headers()
    assert headers["KnowledgatorAPI-Key"] == "kg_bearer_token_789"


def test_api_proxy_missing_key_raises_401():
    """Verify 401 ClientError is raised when api_proxy request lacks an API key."""
    config = ServerConfig(auth_type="api_proxy", base_url="https://mcp.example.com")
    ctx = make_mock_context({})

    with pytest.raises(ClientError) as exc_info:
        get_kg_client(server_config=config, ctx=ctx)

    assert exc_info.value.status_code == 401
    assert "Authentication required" in str(exc_info.value)


def test_api_key_mode_env_fallback():
    """Verify self-hosted api_key mode relies on configured static API key."""
    config = ServerConfig(auth_type="api_key", api_key="sk-env-static-key", base_url="https://api.knowledgator.com")
    ctx = make_mock_context({"knowledgatorapi-key": "ignored_header_key"})

    client, server_url, headers = get_client_context(server_config=config, ctx=ctx)
    assert headers["KnowledgatorAPI-Key"] == "sk-env-static-key"


@pytest.mark.asyncio
async def test_api_proxy_alice_bob_concurrency():
    """Verify two concurrent user requests (Alice & Bob) retain isolated API keys with zero leak."""
    config = ServerConfig(auth_type="api_proxy", base_url="https://mcp.example.com")

    ctx_alice = make_mock_context({"knowledgatorapi-key": "key_alice_123"})
    ctx_bob = make_mock_context({"knowledgatorapi-key": "key_bob_456"})

    captured_calls = []

    async def mock_it_async(text, template=None, output_type=None, model_id=None, server_url=None, http_headers=None, timeout_ms=60000):
        # Simulate async work delay to test race condition resistance
        await asyncio.sleep(0.01)
        captured_calls.append({
            "text": text,
            "headers": http_headers.copy() if http_headers else {},
        })
        return {"result": f"response for {text}"}

    with patch("knowledgator_mcp.tools.inference_tools.get_client_context") as mock_get_context:
        # Construct isolated mock SDK client for Alice
        alice_sdk = MagicMock()
        alice_sdk.instruct.it_async = AsyncMock(side_effect=mock_it_async)
        alice_headers = {"KnowledgatorAPI-Key": "key_alice_123"}

        # Construct isolated mock SDK client for Bob
        bob_sdk = MagicMock()
        bob_sdk.instruct.it_async = AsyncMock(side_effect=mock_it_async)
        bob_headers = {"KnowledgatorAPI-Key": "key_bob_456"}

        def side_effect(server_config=None, auth_provider=None, ctx=None):
            req_key = None
            if ctx and getattr(ctx, "request_context", None):
                req_key = ctx.request_context.headers.get("knowledgatorapi-key")
            if req_key == "key_alice_123":
                return alice_sdk, "https://mcp.example.com", alice_headers
            elif req_key == "key_bob_456":
                return bob_sdk, "https://mcp.example.com", bob_headers
            raise ValueError(f"Unexpected ctx: {ctx}")

        mock_get_context.side_effect = side_effect

        # Run concurrent tool calls for Alice and Bob
        alice_task = instruct_inference(text="Alice prompt", ctx=ctx_alice)
        bob_task = instruct_inference(text="Bob prompt", ctx=ctx_bob)

        alice_res, bob_res = await asyncio.gather(alice_task, bob_task)

        assert alice_res == {"result": "response for Alice prompt"}
        assert bob_res == {"result": "response for Bob prompt"}

        # Verify call arguments
        alice_call = next(c for c in captured_calls if c["text"] == "Alice prompt")
        bob_call = next(c for c in captured_calls if c["text"] == "Bob prompt")

        assert alice_call["headers"]["KnowledgatorAPI-Key"] == "key_alice_123"
        assert bob_call["headers"]["KnowledgatorAPI-Key"] == "key_bob_456"
