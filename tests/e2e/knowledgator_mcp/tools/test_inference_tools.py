import json
import pytest
from knowledgator_mcp.tools import inference_tools


@pytest.mark.asyncio
async def test_instruct_inference():
    """Verify instruct_inference MCP tool against live API with multi-entity executives extraction."""
    text = """
    Elon Musk is CEO of Tesla in Austin. Tim Cook leads Apple in Cupertino.
    Satya Nadella runs Microsoft in Redmond.
    """
    template = {
        "name": "",
        "company": "",
        "role": "",
        "location": ""
    }
    result = await inference_tools.instruct_inference(
        text=text,
        template=template,
        output_type="json",
    )
    assert isinstance(result, dict)
    assert "outputs" in result or "parsed" in result


@pytest.mark.asyncio
async def test_comprehend_inference():
    """Verify comprehend_inference MCP tool against live API using tasks."""
    result = await inference_tools.comprehend_inference(
        text="Apple announced a new iPad with advanced artificial intelligence features.",
        tasks={"topic": {"type": "classify", "labels": ["technology", "business", "sports", "entertainment"]}},
    )
    assert isinstance(result, dict)
    assert "outputs" in result


@pytest.mark.asyncio
async def test_extract_inference():
    """Verify extract_inference MCP tool against live API using tasks."""
    result = await inference_tools.extract_inference(
        text="Steve Jobs co-founded Apple Inc. in California.",
        tasks={"entities": {"type": "span", "labels": ["person", "organization", "location"]}},
    )
    assert isinstance(result, dict)
    assert "outputs" in result


@pytest.mark.asyncio
async def test_instruct_inference_france_recording():
    """Verify instruct_inference against live API with France capital query and JSON template."""
    template_str = json.dumps({"summary": "", "key_points": [], "risk_level": ""})
    result = await inference_tools.instruct_inference(
        text="What is the capital of France?",
        template=template_str,
        output_type="json",
    )
    assert isinstance(result, dict)
    assert "outputs" in result


@pytest.mark.asyncio
async def test_open_model_inference_recording():
    """Verify open_model_inference against live API without failure checking 'outputs' in root."""
    body = {
        "input": "Steve Jobs founded Apple Inc. in Cupertino, California.",
        "tasks": {
            "topic": {"type": "classify", "labels": ["technology", "sports", "politics"], "threshold": 0.5},
            "entities": {"type": "span", "labels": ["person", "organization", "location"]},
        },
        "options": {"include_usage": True},
    }
    result = await inference_tools.open_model_inference(
        model_id="gliformer-large-v1",
        data=body,
    )
    assert isinstance(result, dict)
    assert "outputs" in result
