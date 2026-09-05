import json
import pytest
from knowledgator_mcp.tools import inference_tools


@pytest.mark.asyncio
async def test_instruct_inference():
    """Verify instruct_inference MCP tool against live API."""
    template_str = json.dumps({"summary": "", "key_points": [], "risk_level": ""})
    result = await inference_tools.instruct_inference(
        text="What are the specifics of climate at Antarctica?",
        template=template_str,
        output_type="json",
    )
    assert isinstance(result, dict)
    assert "outputs" in result or "parsed" in result or "results" in result


@pytest.mark.asyncio
async def test_comprehend_inference():
    """Verify comprehend_inference MCP tool against live API."""
    result = await inference_tools.comprehend_inference(
        text="Apple announced a new iPad with advanced artificial intelligence features.",
        labels=["technology", "business", "sports", "entertainment"],
    )
    assert isinstance(result, dict)
    assert "results" in result or "outputs" in result or "labels" in result


@pytest.mark.asyncio
async def test_extract_inference():
    """Verify extract_inference MCP tool against live API."""
    result = await inference_tools.extract_inference(
        text="Steve Jobs co-founded Apple Inc. in California.",
        labels=["person", "organization", "location"],
    )
    assert isinstance(result, dict)
    assert "results" in result or "outputs" in result or "entities" in result
