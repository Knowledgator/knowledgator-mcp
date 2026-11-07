from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from knowledgator_mcp.tools import inference_tools


@pytest.fixture
def mock_client_context():
    mock_client = MagicMock()
    mock_client.comprehend = MagicMock()
    mock_client.extract = MagicMock()
    mock_client.instruct = MagicMock()
    mock_client.model = MagicMock()
    return mock_client, "https://api.test.com", {"KnowledgatorAPI-Key": "test-key"}


@pytest.mark.asyncio
async def test_comprehend_inference_with_adapter_id(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "results": [{"label": "tech", "score": 0.99}],
        "input_text": "Sample text",
        "labels": ["tech", "sports"],
    }
    mock_client.comprehend.it_async = AsyncMock(return_value=mock_resp)

    with patch("knowledgator_mcp.tools.inference_tools.get_client_context", return_value=(mock_client, server_url, headers)):
        res = await inference_tools.comprehend_inference(
            text="Sample text",
            labels=["tech", "sports"],
            adapter_id="adapter-lora-1",
        )

    assert res["results"][0]["label"] == "tech"
    mock_client.comprehend.it_async.assert_awaited_once_with(
        text="Sample text",
        labels=["tech", "sports"],
        threshold=None,
        adapter_id="adapter-lora-1",
        server_url=server_url,
        http_headers=headers,
        timeout_ms=60000,
    )


@pytest.mark.asyncio
async def test_comprehend_inference_legacy_model_id_fallback(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {"results": [{"label": "sports", "score": 0.95}]}
    mock_client.comprehend.it_async = AsyncMock(return_value=mock_resp)

    with patch("knowledgator_mcp.tools.inference_tools.get_client_context", return_value=(mock_client, server_url, headers)):
        res = await inference_tools.comprehend_inference(
            text="Sample text",
            labels=["tech", "sports"],
            adapter_id="adapter-comprehend-1",
        )

    assert res["results"][0]["label"] == "sports"
    mock_client.comprehend.it_async.assert_awaited_once_with(
        text="Sample text",
        labels=["tech", "sports"],
        threshold=None,
        adapter_id="adapter-comprehend-1",
        server_url=server_url,
        http_headers=headers,
        timeout_ms=60000,
    )


@pytest.mark.asyncio
async def test_extract_inference_with_adapter_id(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "results": [{"span": "Apple", "entity": "ORG", "score": 0.98}],
    }
    mock_client.extract.it_async = AsyncMock(return_value=mock_resp)

    with patch("knowledgator_mcp.tools.inference_tools.get_client_context", return_value=(mock_client, server_url, headers)):
        res = await inference_tools.extract_inference(
            text="Apple announced a new product.",
            labels=["ORG"],
            adapter_id="adapter-extract-1",
        )

    assert res["results"][0]["span"] == "Apple"
    mock_client.extract.it_async.assert_awaited_once_with(
        text="Apple announced a new product.",
        labels=["ORG"],
        threshold=None,
        adapter_id="adapter-extract-1",
        server_url=server_url,
        http_headers=headers,
        timeout_ms=60000,
    )


@pytest.mark.asyncio
async def test_instruct_inference_with_adapter_id(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {"outputs": "Extracted summary"}
    mock_client.instruct.it_async = AsyncMock(return_value=mock_resp)

    with patch("knowledgator_mcp.tools.inference_tools.get_client_context", return_value=(mock_client, server_url, headers)):
        res = await inference_tools.instruct_inference(
            text="Summarize this text",
            adapter_id="adapter-instruct-1",
        )

    assert res["outputs"] == "Extracted summary"
    mock_client.instruct.it_async.assert_awaited_once_with(
        text="Summarize this text",
        template=None,
        output_type=None,
        adapter_id="adapter-instruct-1",
        server_url=server_url,
        http_headers=headers,
        timeout_ms=60000,
    )


@pytest.mark.asyncio
async def test_open_model_inference_predict(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_open_model = MagicMock()
    mock_open_model.predict_async = AsyncMock(return_value={"outputs": [{"label": "Clothing", "score": 0.99}]})
    mock_client.model.return_value = mock_open_model

    payload = {"text": "Organic Cotton Hoodie", "labels": ["Clothing", "Home Goods"]}

    with patch("knowledgator_mcp.tools.inference_tools.get_client_context", return_value=(mock_client, server_url, headers)):
        res = await inference_tools.open_model_inference(
            model_id="m-d9be7ede307943439a30c6f814026566",
            data=payload,
            timeout_ms=90000,
        )

    assert res["outputs"][0]["label"] == "Clothing"
    mock_client.model.assert_called_once_with(model_id="m-d9be7ede307943439a30c6f814026566", method=None)
    mock_open_model.predict_async.assert_awaited_once_with(body=payload, timeout_ms=90000)


@pytest.mark.asyncio
async def test_open_model_inference_call_method(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_open_model = MagicMock()
    mock_open_model.call_async = AsyncMock(return_value={"outputs": [{"label": "Electronics", "score": 0.98}]})
    mock_client.model.return_value = mock_open_model

    payload = {"text": "Smart LED Lamp", "labels": ["Electronics", "Home Goods"]}

    with patch("knowledgator_mcp.tools.inference_tools.get_client_context", return_value=(mock_client, server_url, headers)):
        res = await inference_tools.open_model_inference(
            model_id="m-829f9cf569b24bb1b9750c2151dbfc6d",
            data=payload,
            method="classify",
            timeout_ms=45000,
        )

    assert res["outputs"][0]["label"] == "Electronics"
    mock_client.model.assert_called_once_with(model_id="m-829f9cf569b24bb1b9750c2151dbfc6d", method="classify")
    mock_open_model.call_async.assert_awaited_once_with(method="classify", body=payload, timeout_ms=45000)
