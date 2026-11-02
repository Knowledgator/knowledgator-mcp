import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from knowledgator_mcp.tools.dataset_tools import (
    _sanitize_dataset_name,
    _resolve_dest_path,
    _sanitize_filename,
    _validate_classification_dataset,
    _validate_dataset_payload,
    download_dataset,
    upload_dataset,
)


def test_sanitize_filename_edge_cases():
    assert _sanitize_filename("ValidDatasetName") == "ValidDatasetName"
    assert _sanitize_filename("Dataset With Spaces") == "Dataset_With_Spaces"
    assert _sanitize_filename('Dangerous/Path\\With:Bad*Chars?And"Quotes<Or>Pipes|') == "Dangerous_Path_With_Bad_Chars_And_Quotes_Or_Pipes"
    assert _sanitize_filename("...dots-and-dashes...") == "dots-and-dashes"
    assert _sanitize_filename("   \t\n   ") == "dataset"


def test_sanitize_dataset_name_rules():
    assert _sanitize_dataset_name("Products") == "Products"
    assert _sanitize_dataset_name("Products v1.0") == "Products v1.0"
    assert _sanitize_dataset_name("Classification v2") == "Classification v2"
    assert _sanitize_dataset_name("Dataset v2.3.1") == "Dataset v2.3.1"
    assert _sanitize_dataset_name("", fallback="Fallback") == "Fallback"
    # Special character sanitization for OpenAPI pattern ^[a-zA-Z0-9- .()]+$
    res = _sanitize_dataset_name("Bad@Name#Test v1.0")
    assert "@" not in res and "#" not in res
    assert res == "Bad-Name-Test v1.0"


def test_validate_classification_dataset_valid_and_invalid():
    valid = [
        {"id": "1", "text": "Customer love this phone", "label": "electronics"},
        {"content": "Great cotton shirt", "category": "apparel"},
        {"sentence": "Fresh organic strawberries", "target": "grocery"},
        {
            "text": "This coffee never disappoints.",
            "true_labels": ["positive"],
            "all_labels": ["negative", "positive"],
        },
    ]
    assert len(_validate_classification_dataset(valid)) == 4

    # Rejection cases
    with pytest.raises(ValueError, match="must be a JSON list"):
        _validate_classification_dataset({"text": "test", "label": "test"})

    with pytest.raises(ValueError, match="is empty"):
        _validate_classification_dataset([])

    with pytest.raises(ValueError, match="must be a dictionary"):
        _validate_classification_dataset(["not-a-dict"])

    with pytest.raises(ValueError, match="missing a valid text field"):
        _validate_classification_dataset([{"label": "tech"}])

    with pytest.raises(ValueError, match="missing a classification target/label"):
        _validate_classification_dataset([{"text": "Only text without label"}])


def test_validate_dataset_payload_dict_wrapper():
    wrapped = {
        "data": [
            {"id": "1", "text": "Sample text", "label": "test_label"}
        ]
    }
    validated = _validate_dataset_payload(wrapped, dataset_type="comprehend-it")
    assert len(validated) == 1
    assert validated[0]["text"] == "Sample text"


@pytest.mark.asyncio
async def test_upload_dataset_from_local_file(tmp_path):
    dataset_file = tmp_path / "test_samples.json"
    samples = [
        {"id": "s1", "text": "Super fast delivery", "label": "shipping"},
        {"id": "s2", "text": "Broken glass in box", "label": "damage"},
    ]
    dataset_file.write_text(json.dumps(samples), encoding="utf-8")

    with patch("knowledgator_mcp.tools.dataset_tools.get_client_context") as mock_ctx:
        mock_client = MagicMock()
        mock_create = AsyncMock()
        mock_create.return_value = {
            "id": "6babd48a-2256-4466-b16b-9fd45ba59688",
            "status": "ready",
            "name": "Shipping Reviews",
        }
        mock_client.datasets.create_async = mock_create
        mock_ctx.return_value = (mock_client, "https://api.knowledgator.com", {"header": "val"})

        result = await upload_dataset(
            name="Shipping Reviews",
            json_content=dataset_file.read_text(encoding="utf-8"),
            dataset_type="classification",
            short_description="Audit feedback dataset",
        )

        assert result["status"] == "success"
        assert result["dataset_id"] == "6babd48a-2256-4466-b16b-9fd45ba59688"
        assert result["dataset_name"] == "Shipping Reviews"
        assert result["sample_count"] == 2

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["name"] == "Shipping Reviews"
        assert call_kwargs["type"] == "comprehend-it"
        assert call_kwargs["format"] == "json"
        assert call_kwargs["short_description"] == "Audit feedback dataset"
        assert json.loads(call_kwargs["file"].decode("utf-8")) == samples


@pytest.mark.asyncio
async def test_upload_dataset_from_json_string():
    samples = [{"text": "Great service", "label": "positive"}]
    json_str = json.dumps(samples)

    with patch("knowledgator_mcp.tools.dataset_tools.get_client_context") as mock_ctx:
        mock_client = MagicMock()
        mock_create = AsyncMock()
        mock_create.return_value = {
            "id": "78d379da228a4985ba19100f53245b79",
            "status": "ready",
        }
        mock_client.datasets.create_async = mock_create
        mock_ctx.return_value = (mock_client, "https://api.knowledgator.com", {})

        result = await upload_dataset(
            name="String Test v2",
            json_content=json_str,
            dataset_type="comprehend-it",
        )

        assert result["status"] == "success"
        assert result["dataset_id"] == "78d379da228a4985ba19100f53245b79"
        assert result["sample_count"] == 1


@pytest.mark.asyncio
async def test_upload_dataset_validation_failures(tmp_path):
    # Missing all sources
    with pytest.raises(ValueError, match="Either 'data' or 'json_content' must be provided."):
        await upload_dataset(name="Empty")

    # Invalid JSON string
    with pytest.raises(ValueError, match="Invalid JSON provided in json_content"):
        await upload_dataset(name="BadJSON", json_content="{not valid json}")


@pytest.mark.asyncio
async def test_download_dataset_stateless_in_memory():
    samples = [
        {"id": "ex_1", "text": "Sample 1", "label": "cat_a"},
        {"id": "ex_2", "text": "Sample 2", "label": "cat_b"},
    ]

    with patch("knowledgator_mcp.tools.dataset_tools.get_client_context") as mock_ctx:
        mock_client = MagicMock()
        mock_client.datasets.download_async = AsyncMock(return_value=samples)
        mock_ctx.return_value = (mock_client, "https://api.knowledgator.com", {"header": "val"})

        res = await download_dataset(dataset_id="stateless-uuid-123")
        assert res == samples
        mock_client.datasets.download_async.assert_called_once()


@pytest.mark.asyncio
async def test_get_dataset_details_metadata():
    with patch("knowledgator_mcp.tools.dataset_tools.get_client_context") as mock_ctx:
        mock_client = MagicMock()
        mock_client.datasets.get.return_value = {"id": "disk-ds-uuid", "name": "DiskDataset"}
        mock_ctx.return_value = (mock_client, "https://api.knowledgator.com", {})

        from knowledgator_mcp.tools.dataset_tools import get_dataset_details
        res = await get_dataset_details(dataset_id="disk-ds-uuid")

        assert res["id"] == "disk-ds-uuid"
        assert res["name"] == "DiskDataset"


@pytest.mark.asyncio
async def test_download_dataset_streaming_to_file(tmp_path):
    dest_file = tmp_path / "streamed.json"
    chunk_1 = b'[{"text": "Sample", '
    chunk_2 = b'"label": "test"}]'

    async def fake_stream(*args, **kwargs):
        yield chunk_1
        yield chunk_2

    with patch("knowledgator_mcp.tools.dataset_tools.get_client_context") as mock_ctx:
        mock_client = MagicMock(spec=["datasets"])
        mock_datasets = MagicMock(spec=["stream_download_async"])
        mock_datasets.stream_download_async = fake_stream
        mock_client.datasets = mock_datasets
        mock_ctx.return_value = (mock_client, "https://api.knowledgator.com", {})

        res = await download_dataset(
            dataset_id="stream-uuid-123",
        )

        assert res == [{"text": "Sample", "label": "test"}]


@pytest.mark.asyncio
async def test_upload_dataset_stateless_pass_through():
    samples = [{"text": "Stream text", "label": "label_a"}]
    json_str = json.dumps(samples)

    with patch("knowledgator_mcp.tools.dataset_tools.get_client_context") as mock_ctx:
        mock_client = MagicMock()
        mock_create = AsyncMock()
        mock_create.return_value = {
            "id": "streamed-upload-id-123",
            "status": "ready",
            "name": "Stateless Upload",
        }
        mock_client.datasets.create_async = mock_create
        mock_ctx.return_value = (mock_client, "https://api.knowledgator.com", {})

        res = await upload_dataset(
            name="Stateless Upload",
            json_content=json_str,
            dataset_type="classification",
        )

        assert res["status"] == "success"
        assert res["dataset_id"] == "streamed-upload-id-123"
        assert res["sample_count"] == 1
        call_kwargs = mock_create.call_args.kwargs
        assert "file" in call_kwargs
        assert json.loads(call_kwargs["file"].decode("utf-8")) == samples
