import pytest
from knowledgator_mcp.tools import dataset_tools


@pytest.mark.asyncio
async def test_list_datasets():
    """Verify list_datasets MCP tool against live API."""
    result = await dataset_tools.list_datasets(limit=5)
    assert isinstance(result, (dict, list))


@pytest.mark.asyncio
async def test_get_dataset_details_existing():
    """Verify get_dataset_details with an existing dataset ID if present."""
    result = await dataset_tools.list_datasets(limit=5)
    datasets = []
    if isinstance(result, dict):
        datasets = result.get("datasets") or result.get("data") or []
    elif isinstance(result, list):
        datasets = result

    if datasets and isinstance(datasets[0], dict) and "id" in datasets[0]:
        dataset_id = datasets[0]["id"]
        details_res = await dataset_tools.get_dataset_details(dataset_id)
        assert isinstance(details_res, (dict, list))


@pytest.mark.asyncio
async def test_generate_dataset_plan():
    """Verify generate_dataset_plan MCP tool against live API."""
    result = await dataset_tools.generate_dataset_plan(
        prompt="Create a sentiment analysis dataset for product reviews",
    )
    assert isinstance(result, (dict, list))


@pytest.mark.asyncio
async def test_start_dataset_generation():
    """Verify start_dataset_generation MCP tool against live API."""
    result = await dataset_tools.start_dataset_generation(
        name="test-e2e-gen-dataset",
        dataset_type="text-classification",
        prompt="Product review sentiment classification",
        description="Dataset for product review sentiment classification",
        num_samples=1,
        classes=["positive", "negative"],
        domains=["e-commerce"],
    )
    assert isinstance(result, (dict, list))
    print(f"{result=}")



