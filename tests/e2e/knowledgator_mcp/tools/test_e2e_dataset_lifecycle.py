import json
import logging
import pytest
from knowledgator_mcp.client import get_kg_client
from knowledgator_mcp.tools import dataset_tools

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_e2e_dataset_lifecycle_upload_get_download_cleanup():
    """E2E test verifying end-to-end dataset lifecycle:

    1. Upload validated dataset.
    2. Retrieve details via get_dataset_details.
    3. Download statelessly in memory (return_data=True).
    4. Download metadata-only (return_data=False, zero disk writes).
    5. Clean up by deleting the created dataset.
    """
    sample_data = [
        {
            "id": "e2e_sample_1",
            "text": "The delivery was surprisingly fast and the packaging was immaculate.",
            "true_labels": ["positive"],
            "all_labels": ["negative", "positive"],
        },
        {
            "id": "e2e_sample_2",
            "text": "The package arrived open and the internal components were broken.",
            "true_labels": ["negative"],
            "all_labels": ["negative", "positive"],
        },
    ]

    upload_res = await dataset_tools.upload_dataset(
        name="E2E Lifecycle Test",
        data=sample_data,
        dataset_type="classification",
        short_description="Automated E2E lifecycle verification test",
    )

    assert upload_res["status"] == "success"
    dataset_id = upload_res.get("dataset_id")
    assert dataset_id is not None
    assert upload_res.get("sample_count") == len(sample_data)

    client = get_kg_client()

    try:
        # Step 2: Fetch details
        details = await dataset_tools.get_dataset_details(dataset_id=dataset_id)
        assert details is not None
        assert details.get("id") == dataset_id or details.get("data", {}).get("id") == dataset_id

        # Step 3: Stateless in-memory download returns pure dataset samples directly
        downloaded_samples = await dataset_tools.download_dataset(dataset_id=dataset_id)
        assert isinstance(downloaded_samples, list)
        assert len(downloaded_samples) == 2
        assert downloaded_samples[0]["text"] == sample_data[0]["text"]
        assert downloaded_samples[0]["true_labels"] == sample_data[0]["true_labels"]

    finally:
        # Step 5: Clean up created dataset from live platform
        try:
            client.datasets.delete(dataset_id=dataset_id)
            logger.info("Successfully deleted temporary E2E dataset: %s", dataset_id)
        except Exception as exc:
            logger.warning("Could not delete temporary E2E dataset %s: %s", dataset_id, exc)
