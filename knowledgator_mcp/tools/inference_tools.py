from typing import Any, Dict, List, Optional

from knowledgator_mcp.client import get_client_context
from knowledgator_mcp.utils import serialize


async def instruct_inference(
    text: str,
    template: Optional[str] = None,
    output_type: Optional[str] = None,
    model_id: Optional[str] = None,
    timeout_ms: Optional[int] = 60000,
) -> Dict[str, Any]:
    """Executes instruction task using Knowledgator Instruct API.

    Args:
        text: Main prompt instruction or text query.
        template: Optional JSON schema template string the output should follow.
        output_type: Optional hint for desired output format (json, string, markdown, etc.).
        model_id: Optional model identifier override.
        timeout_ms: Optional request timeout in milliseconds (default: 60000 ms = 60s).

    Returns:
        Structured instruct result output.
    """
    client, server_url, headers = get_client_context()
    response = await client.instruct.it_async(
        text=text,
        template=template,
        output_type=output_type,
        model_id=model_id,
        server_url=server_url,
        http_headers=headers,
        timeout_ms=timeout_ms,
    )
    return serialize(response)


async def extract_inference(
    text: str,
    labels: List[str],
    threshold: Optional[float] = None,
    model_id: Optional[str] = None,
    timeout_ms: Optional[int] = 60000,
) -> Dict[str, Any]:
    """Executes entity/relation extraction on text.

    Args:
        text: Target input text to extract entities or relations from.
        labels: List of target entity/relation labels to extract.
        threshold: Optional confidence threshold float (0.0 to 1.0).
        model_id: Optional model identifier override.
        timeout_ms: Optional request timeout in milliseconds (default: 60000 ms = 60s).

    Returns:
        Extraction results with detected entities and spans.
    """
    client, server_url, headers = get_client_context()
    response = await client.extract.it_async(
        text=text,
        labels=labels,
        threshold=threshold,
        model_id=model_id,
        server_url=server_url,
        http_headers=headers,
        timeout_ms=timeout_ms,
    )
    return serialize(response)


async def comprehend_inference(
    text: str,
    labels: List[str],
    threshold: Optional[float] = None,
    model_id: Optional[str] = None,
    timeout_ms: Optional[int] = 60000,
) -> Dict[str, Any]:
    """Executes text classification and zero-shot comprehension.

    Args:
        text: Input text to categorize or comprehend.
        labels: List of candidate classification labels.
        threshold: Optional confidence threshold (0.0 to 1.0).
        model_id: Optional model identifier override.
        timeout_ms: Optional request timeout in milliseconds (default: 60000 ms = 60s).

    Returns:
        Classification results with label confidence scores.
    """
    client, server_url, headers = get_client_context()
    response = await client.comprehend.it_async(
        text=text,
        labels=labels,
        threshold=threshold,
        model_id=model_id,
        server_url=server_url,
        http_headers=headers,
        timeout_ms=timeout_ms,
    )
    return serialize(response)
