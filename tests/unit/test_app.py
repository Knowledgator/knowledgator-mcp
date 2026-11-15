import pytest

from knowledgator_mcp.app import mcp


@pytest.mark.asyncio
async def test_mcp_tools_registration():
    tools = await mcp.list_tools()
    tool_names = [tool.name for tool in tools]

    # Check Dataset tools
    assert "generate_dataset_plan" in tool_names
    assert "start_dataset_generation" in tool_names
    assert "list_datasets" in tool_names
    assert "get_dataset_details" in tool_names
    assert "get_dataset_analytics_report" in tool_names
    assert "list_data" in tool_names
    assert "download_dataset" in tool_names
    assert "upload_dataset" in tool_names

    # Check Fine-Tuning tools
    assert "start_training_job" in tool_names
    assert "get_training_status" in tool_names
    assert "list_trainings" in tool_names
    assert "list_models" in tool_names
    assert "stop_training_job" in tool_names
    assert "get_model_details" in tool_names
    assert "get_inference_details" in tool_names
    assert "get_training_metrics_report" in tool_names

    # Check Inference tools
    assert "instruct_inference" in tool_names
    assert "extract_inference" in tool_names
    assert "comprehend_inference" in tool_names
    assert "open_model_inference" in tool_names


@pytest.mark.asyncio
async def test_mcp_resources_registration():
    resources = await mcp.list_resources()
    resource_uris = [str(r.uri) for r in resources]
    assert "config://app" in resource_uris


def test_mcp_transport_security_configured():
    assert mcp.settings.transport_security is not None
    assert mcp.settings.transport_security.enable_dns_rebinding_protection is True
    assert "localhost" in mcp.settings.transport_security.allowed_hosts


@pytest.mark.asyncio
async def test_streamable_http_graceful_cleanup(caplog):
    import anyio
    import logging
    from unittest.mock import AsyncMock, MagicMock
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    transport = MagicMock(spec=StreamableHTTPServerTransport)
    transport._request_streams = {}

    send_mock = AsyncMock()
    recv_mock = AsyncMock()
    req_id = "test-session-123"
    transport._request_streams[req_id] = (send_mock, recv_mock)

    await StreamableHTTPServerTransport._clean_up_memory_streams(transport, req_id)

    # Verify both streams were closed and removed from dictionary
    send_mock.aclose.assert_awaited_once()
    recv_mock.aclose.assert_awaited_once()
    assert req_id not in transport._request_streams

    # Verify unexpected errors during close are logged at ERROR level
    error_send_mock = AsyncMock()
    error_send_mock.aclose.side_effect = RuntimeError("Failed to close send stream")
    error_recv_mock = AsyncMock()
    error_recv_mock.aclose.side_effect = RuntimeError("Failed to close recv stream")
    error_req_id = "error-session-456"
    transport._request_streams[error_req_id] = (error_send_mock, error_recv_mock)

    with caplog.at_level(logging.ERROR):
        await StreamableHTTPServerTransport._clean_up_memory_streams(transport, error_req_id)

    assert "Error closing send stream for request error-session-456" in caplog.text
    assert "Error closing recv stream for request error-session-456" in caplog.text

