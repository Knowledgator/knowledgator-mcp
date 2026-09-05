import os

import pytest
from dotenv import load_dotenv

try:
    load_dotenv(override=True)
except ImportError:
    pass

API_KEY = os.getenv("KG_MCP_API_KEY") or os.getenv("KG_API_KEY")


def pytest_collection_modifyitems(config, items):
    """Skip E2E test items if no API key is set in environment."""
    if not API_KEY:
        skip_marker = pytest.mark.skip(
            reason="KG_MCP_API_KEY or KG_API_KEY environment variable is required to run E2E live API tests"
        )
        for item in items:
            item.add_marker(skip_marker)


@pytest.fixture(autouse=True)
def setup_fresh_client():
    from knowledgator_mcp.client import close_kg_client
    close_kg_client()
    yield
    close_kg_client()


@pytest.fixture
def sync_sdk_client():
    from knowledgator_mcp.client import get_kg_client
    return get_kg_client(force_refresh=True)
