from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from knowledgator_mcp.tools import finetuning_tools


@pytest.fixture
def mock_client_context():
    mock_client = MagicMock()
    mock_client.finetuning = MagicMock()
    return mock_client, "https://api.test.com", {"KnowledgatorAPI-Key": "test-key-123"}


@pytest.mark.asyncio
async def test_list_models_all(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "total": 2,
        "models": [
            {
                "id": "gliclass-v4-2b",
                "name": "gliclass-v4-2b",
                "type": "GLiClass",
                "fine_tuned": False,
                "training": {"model": False, "adapter": False},
            },
            {
                "id": "instruct-it",
                "name": "instruct-it",
                "type": "GLiFormer",
                "fine_tuned": False,
                "training": {"model": False, "adapter": True},
            },
        ],
    }
    mock_client.finetuning.list_models_async = AsyncMock(return_value=mock_resp)

    with patch(
        "knowledgator_mcp.tools.finetuning_tools.get_client_context",
        return_value=(mock_client, server_url, headers),
    ):
        res = await finetuning_tools.list_models(offset=0, limit=20)

    assert res["total"] == 2
    assert len(res["models"]) == 2
    assert res["models"][0]["id"] == "gliclass-v4-2b"
    assert "url" in res["models"][0]
    assert res["models"][0]["url"] == "https://platform.knowledgator.com/models/gliclass-v4-2b"

    mock_client.finetuning.list_models_async.assert_awaited_once_with(
        offset=0,
        limit=20,
        object_type=None,
        trainable=None,
        inference=None,
        type=None,
        name=None,
        server_url=server_url,
        http_headers=headers,
    )


@pytest.mark.asyncio
async def test_list_models_filter_adapters(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "total": 1,
        "models": [
            {
                "id": "custom-adapter-1",
                "name": "custom-adapter-1",
                "type": "GLiClass",
                "fine_tuned": True,
                "parent_model": "gliclass-v4-2b",
                "training": {"model": False, "adapter": True},
            }
        ],
    }
    mock_client.finetuning.list_models_async = AsyncMock(return_value=mock_resp)

    with patch(
        "knowledgator_mcp.tools.finetuning_tools.get_client_context",
        return_value=(mock_client, server_url, headers),
    ):
        res = await finetuning_tools.list_models(
            object_type="adapter",
            trainable="adapter",
            inference="comprehend-it",
            type="GLiClass",
            name="custom",
            offset=10,
            limit=5,
        )

    assert res["total"] == 1
    assert res["models"][0]["id"] == "custom-adapter-1"
    assert res["models"][0]["url"] == "https://platform.knowledgator.com/models/custom-adapter-1"

    mock_client.finetuning.list_models_async.assert_awaited_once_with(
        offset=10,
        limit=5,
        object_type="adapter",
        trainable="adapter",
        inference="comprehend-it",
        type="GLiClass",
        name="custom",
        server_url=server_url,
        http_headers=headers,
    )


@pytest.mark.asyncio
async def test_list_models_filter_full_models(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "total": 1,
        "models": [
            {
                "id": "gliner-multitask-v1-0",
                "name": "gliner-multitask-v1-0",
                "type": "GLiNER",
                "fine_tuned": False,
                "training": {"model": True, "adapter": False},
            }
        ],
    }
    mock_client.finetuning.list_models_async = AsyncMock(return_value=mock_resp)

    with patch(
        "knowledgator_mcp.tools.finetuning_tools.get_client_context",
        return_value=(mock_client, server_url, headers),
    ):
        res = await finetuning_tools.list_models(
            object_type="model",
            trainable="model",
            inference="open-models",
        )

    assert res["total"] == 1
    assert res["models"][0]["id"] == "gliner-multitask-v1-0"
    assert res["models"][0]["url"] == "https://platform.knowledgator.com/models/gliner-multitask-v1-0"

    mock_client.finetuning.list_models_async.assert_awaited_once_with(
        offset=0,
        limit=20,
        object_type="model",
        trainable="model",
        inference="open-models",
        type=None,
        name=None,
        server_url=server_url,
        http_headers=headers,
    )


@pytest.mark.asyncio
async def test_list_models_finetuning_disabled():
    mock_client = MagicMock()
    mock_client.finetuning = None

    with patch(
        "knowledgator_mcp.tools.finetuning_tools.get_client_context",
        return_value=(mock_client, "https://api.test.com", {}),
    ):
        with pytest.raises(RuntimeError, match="Finetuning API is not enabled"):
            await finetuning_tools.list_models()


@pytest.mark.asyncio
async def test_list_models_description_compactness(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "total": 1,
        "models": [
            {
                "id": "model-large-desc",
                "name": "model-large-desc",
                "short_description": "Compact summary",
                "description": "# Massive 100KB README with tables and code",
                "type": "GLiClass",
            }
        ],
    }
    mock_client.finetuning.list_models_async = AsyncMock(return_value=mock_resp)

    # 1. Default: include_description=False -> description stripped to save payload
    with patch(
        "knowledgator_mcp.tools.finetuning_tools.get_client_context",
        return_value=(mock_client, server_url, headers),
    ):
        res_compact = await finetuning_tools.list_models()
    assert "description" not in res_compact["models"][0]
    assert res_compact["models"][0]["short_description"] == "Compact summary"

    # 2. include_description=True -> description preserved
    mock_resp.model_dump.return_value = {
        "total": 1,
        "models": [
            {
                "id": "model-large-desc",
                "name": "model-large-desc",
                "short_description": "Compact summary",
                "description": "# Massive 100KB README with tables and code",
                "type": "GLiClass",
            }
        ],
    }
    with patch(
        "knowledgator_mcp.tools.finetuning_tools.get_client_context",
        return_value=(mock_client, server_url, headers),
    ):
        res_full = await finetuning_tools.list_models(include_description=True)
    assert res_full["models"][0]["description"] == "# Massive 100KB README with tables and code"


@pytest.mark.asyncio
async def test_list_trainings_tool(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "total": 1,
        "trainings": [
            {
                "id": "train-123",
                "type": "comprehend-it",
                "status": "completed",
                "name": "my-fine-tuned-model",
            }
        ],
    }
    mock_client.finetuning.list_trainings_async = AsyncMock(return_value=mock_resp)

    with patch(
        "knowledgator_mcp.tools.finetuning_tools.get_client_context",
        return_value=(mock_client, server_url, headers),
    ):
        res = await finetuning_tools.list_trainings(
            status="completed",
            type="comprehend-it",
            limit=50,
            offset=0,
        )

    assert res["total"] == 1
    assert res["trainings"][0]["id"] == "train-123"
    assert "url" in res["trainings"][0]
    mock_client.finetuning.list_trainings_async.assert_awaited_once_with(
        offset=0,
        limit=50,
        model_id=None,
        dataset_id=None,
        status="completed",
        type="comprehend-it",
        server_url=server_url,
        http_headers=headers,
    )


@pytest.mark.asyncio
async def test_get_model_details_async(mock_client_context):
    mock_client, server_url, headers = mock_client_context
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "id": "gliclass-small-v1-0",
        "name": "gliclass-small-v1-0",
        "description": "Full README content",
    }
    mock_client.finetuning.get_model_async = AsyncMock(return_value=mock_resp)

    with patch(
        "knowledgator_mcp.tools.finetuning_tools.get_client_context",
        return_value=(mock_client, server_url, headers),
    ):
        ctx = MagicMock()
        res = await finetuning_tools.get_model_details(ctx=ctx, model_id="gliclass-small-v1-0")

    assert res["id"] == "gliclass-small-v1-0"
    assert res["description"] == "Full README content"
    assert res["url"] == "https://platform.knowledgator.com/models/gliclass-small-v1-0"
    mock_client.finetuning.get_model_async.assert_awaited_once_with(
        model_id="gliclass-small-v1-0",
        server_url=server_url,
        http_headers=headers,
    )
