import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from knowledgator_mcp.tools.dataset_tools import (
    _sanitize_dataset_name,
    _validate_classification_dataset,
    download_dataset,
    upload_dataset,
)


def test_sanitize_dataset_name():
    assert _sanitize_dataset_name("My Dataset") == "My Dataset"
    assert _sanitize_dataset_name("My Dataset v1.0") == "My Dataset v1.0"
    assert _sanitize_dataset_name("Classification v2") == "Classification v2"
    assert _sanitize_dataset_name("Dataset v2.3.1") == "Dataset v2.3.1"
    assert _sanitize_dataset_name("", fallback="Products") == "Products"
    # Special character sanitization for OpenAPI pattern ^[a-zA-Z0-9- .()]+$
    res = _sanitize_dataset_name("Bad@Name#Test")
    assert "@" not in res and "#" not in res
    assert res == "Bad-Name-Test"


def test_validate_classification_dataset_valid():
    valid_data = [
        {"id": "1", "text": "Great laptop with fast CPU", "label": "electronics"},
        {"id": "2", "content": "Organic apple juice", "category": "grocery"},
    ]
    res = _validate_classification_dataset(valid_data)
    assert len(res) == 2


def test_validate_classification_dataset_rejects_garbage():
    # Not a list
    with pytest.raises(ValueError, match="must be a JSON list"):
        _validate_classification_dataset({"text": "hello", "label": "greeting"})

    # Empty list
    with pytest.raises(ValueError, match="is empty"):
        _validate_classification_dataset([])

    # List of non-dict
    with pytest.raises(ValueError, match="must be a dictionary"):
        _validate_classification_dataset(["just a string"])

    # Missing text field
    with pytest.raises(ValueError, match="missing a valid text field"):
        _validate_classification_dataset([{"label": "tech"}])

    # Missing label/category field
    with pytest.raises(ValueError, match="missing a classification target/label"):
        _validate_classification_dataset([{"text": "Some random text without class"}])


@pytest.mark.asyncio
async def test_download_dataset_stateless_in_memory():
    mock_samples = [
        {"id": "ex_1", "text": "Sample 1", "label": "cat_a"},
        {"id": "ex_2", "text": "Sample 2", "label": "cat_b"},
    ]

    with patch("knowledgator_mcp.tools.dataset_tools.get_client_context") as mock_ctx:
        mock_client = MagicMock()
        mock_client.datasets.download_async = AsyncMock(return_value=mock_samples)
        mock_ctx.return_value = (mock_client, "https://api.test.com", {"header": "val"})

        # Call without file_path -> returns pure samples directly in-memory
        res = await download_dataset(dataset_id="stateless-uuid-123")
        assert res == mock_samples
        mock_client.datasets.download_async.assert_called_once()


@pytest.mark.asyncio
async def test_upload_dataset_stateless_flow():
    mock_samples = [
        {"id": "ex_1", "text": "Audited text", "label": "cat_a"},
    ]

    with patch("knowledgator_mcp.tools.dataset_tools.get_client_context") as mock_ctx:
        mock_client = MagicMock()
        mock_create = AsyncMock()
        mock_create.return_value = {"id": "new-dataset-uuid-999", "name": "Products"}
        mock_client.datasets.create_async = mock_create
        mock_ctx.return_value = (mock_client, "https://api.test.com", {"header": "val"})

        res = await upload_dataset(
            name="Products",
            data=mock_samples,
            dataset_type="classification",
            short_description="Fixed bad examples",
        )

        assert res["status"] == "success"
        assert res["dataset_id"] == "new-dataset-uuid-999"
        assert res["dataset_name"] == "Products"
        assert res["sample_count"] == 1

        # Verify SDK was called with in-memory bytes and correct metadata
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["name"] == "Products"
        assert call_kwargs["type"] == "comprehend-it"
        assert call_kwargs["format"] == "json"
        assert call_kwargs["short_description"] == "Fixed bad examples"
        assert isinstance(call_kwargs["file"], bytes)
        uploaded_data = json.loads(call_kwargs["file"].decode("utf-8"))
        assert uploaded_data == mock_samples
