import json
import logging
import pytest
from knowledgator_mcp.client import get_kg_client
from knowledgator_mcp.tools import dataset_tools

logger = logging.getLogger(__name__)

REAL_DATASET_ID = "d655eda9-ca67-4a86-a8bc-5b3911233c79"


def _debug_print_samples(title: str, parsed: any, file_size: int):
    """Debug output helper to display dataset sample info (first, middle, last)."""
    print(f"\n{'='*25} DEBUG: {title} {'='*25}")
    print(f"Data Size: {file_size} bytes")

    items = []
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
        items = parsed["data"]

    total = len(items)
    print(f"Total Samples: {total}")

    if total == 0:
        print(f"Raw Object: {json.dumps(parsed, indent=2)[:500]}")
    elif total == 1:
        print(f"\n[Single Sample (0/1)]:\n{json.dumps(items[0], indent=2)}")
    elif total == 2:
        print(f"\n[First Sample (0/2)]:\n{json.dumps(items[0], indent=2)}")
        print(f"\n[Last Sample (1/2)]:\n{json.dumps(items[1], indent=2)}")
    else:
        mid_idx = total // 2
        print(f"\n[First Sample (0/{total})]:\n{json.dumps(items[0], indent=2)}")
        print(f"\n[Middle Sample ({mid_idx}/{total})]:\n{json.dumps(items[mid_idx], indent=2)}")
        print(f"\n[Last Sample ({total - 1}/{total})]:\n{json.dumps(items[-1], indent=2)}")
    print(f"{'='*65}\n")


@pytest.mark.asyncio
async def test_e2e_mcp_real_dataset_download_stateless():
    """Verify stateless download of real dataset via MCP returns pure samples directly."""
    samples = await dataset_tools.download_dataset(dataset_id=REAL_DATASET_ID)

    assert isinstance(samples, list)
    assert len(samples) >= 1
    assert "text" in samples[0]

    _debug_print_samples("MCP Stateless Download Result (Pure Samples)", samples, len(json.dumps(samples).encode("utf-8")))


@pytest.mark.asyncio
async def test_e2e_mcp_real_dataset_get_metadata():
    """Verify fetching metadata separately via get_dataset_details."""
    details = await dataset_tools.get_dataset_details(dataset_id=REAL_DATASET_ID)

    assert details is not None
    assert details.get("id") == REAL_DATASET_ID or details.get("data", {}).get("id") == REAL_DATASET_ID


@pytest.mark.asyncio
async def test_e2e_mcp_real_dataset_upload_exact_name_and_cleanup():
    """Verify uploading dataset via MCP preserves exact name, then deletes it."""
    exact_name = "e2e-mcp-real-upload"
    sample_data = [
        {
            "text": "This is the third time I've ordered this particular brand of coffee, and it never disappoints. The aroma when you open the bag is incredible, and the taste is consistently smooth with just the right amount of bitterness. I usually get the medium roast, and it's perfect for my morning pour-over. It's a bit pricier than some other brands, but the quality really shines through. Definitely my go-to for a quality cup of coffee.",
            "true_labels": ["positive"],
            "all_labels": ["negative", "positive"],
        }
    ]

    upload_res = await dataset_tools.upload_dataset(
        name=exact_name,
        data=sample_data,
        dataset_type="classification",
        short_description="E2E MCP upload test with exact name preservation",
    )

    assert upload_res["status"] == "success"
    created_id = upload_res.get("dataset_id")
    assert created_id is not None
    assert upload_res.get("sample_count") == len(sample_data)

    assert upload_res.get("dataset_name") == exact_name

    # Cleanup created dataset
    client = get_kg_client()
    try:
        client.datasets.delete(dataset_id=created_id)
        logger.info("Successfully deleted temporary E2E dataset: %s", created_id)
    except Exception as exc:
        logger.warning("Could not delete temporary E2E dataset %s: %s", created_id, exc)
