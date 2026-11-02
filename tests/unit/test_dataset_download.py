import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from knowledgator_mcp.tools.dataset_tools import (
    _resolve_dest_path,
    _sanitize_filename,
    download_dataset,
)


def test_sanitize_filename():
    assert _sanitize_filename("NormalName") == "NormalName"
    assert _sanitize_filename("Name With Spaces") == "Name_With_Spaces"
    assert _sanitize_filename('Bad/Slash\\Back:Colon*Star?Quote"Less<Greater>Pipe|') == "Bad_Slash_Back_Colon_Star_Quote_Less_Greater_Pipe"
    assert _sanitize_filename("...trailing.dots...") == "trailing.dots"
    assert _sanitize_filename("   ") == "dataset"


def test_resolve_dest_path_automatic_from_details():
    dataset_name = "Comprehend-it(Ecomm Item Category Classification Dataset)"
    dest = _resolve_dest_path(dataset_name=dataset_name, output_path=None, output_dir="data")
    assert dest.name == "Comprehend-it(Ecomm_Item_Category_Classification_Dataset).json"
    assert dest.suffix == ".json"


def test_resolve_dest_path_gemini_bad_name(tmp_path):
    # Gemini passes filename with illegal characters and no extension
    bad_name = str(tmp_path / "My:Bad*Dataset?V1.1")
    dest = _resolve_dest_path(dataset_name="Fallback", output_path=bad_name)
    assert dest.name == "My_Bad_Dataset_V1.1.json"
    assert dest.suffix == ".json"
    assert dest.parent == tmp_path


def test_resolve_dest_path_directory_target(tmp_path):
    # User passes a directory path ending in /
    dir_path = str(tmp_path / "custom_dir") + "/"
    dest = _resolve_dest_path(dataset_name="Products List", output_path=dir_path)
    assert dest.name == "Products_List.json"
    assert dest.parent == (tmp_path / "custom_dir")


def test_resolve_dest_path_enforces_json_only(tmp_path):
    # User tries to use .txt or no extension
    file_path = str(tmp_path / "dataset_backup")
    dest = _resolve_dest_path(dataset_name="Ignored", output_path=file_path)
    assert dest.name == "dataset_backup.json"
    assert dest.suffix == ".json"


@pytest.mark.asyncio
async def test_download_dataset_empty_content_raises_error():
    with patch("knowledgator_mcp.tools.dataset_tools.get_client_context") as mock_ctx:
        mock_client = MagicMock()
        mock_client.datasets.download_async = AsyncMock(return_value=[])
        mock_ctx.return_value = (mock_client, "https://api.test.com", {"header": "val"})

        with pytest.raises(RuntimeError, match="empty \\(0 bytes\\)"):
            await download_dataset(
                dataset_id="test-id",
            )
