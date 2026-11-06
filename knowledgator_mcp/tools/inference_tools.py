from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context

from knowledgator_mcp.utils import serialize
from knowledgator_mcp.client import get_client_context
from knowledgator.apis.open_model import OpenModelAPI


async def instruct_inference(
    ctx: Context,
    text: str,
    template: Optional[str] = None,
    output_type: Optional[str] = None,
    model_id: Optional[str] = None,
    timeout_ms: Optional[int] = 60000,
) -> Dict[str, Any]:
    """Instruction task. Use when model's inference='instruct-it' or for default Instruct-It service.

    Args:
        text: Prompt/instruction.
        template: JSON schema template.
        output_type: Output format hint.
        model_id: Model ID, or None for default service.
        timeout_ms: Timeout in ms (default 60000).
        ctx: FastMCP request context.

    Returns:
        Instruct result.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
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
    ctx: Context,
    text: str,
    labels: List[str],
    threshold: Optional[float] = None,
    model_id: Optional[str] = None,
    timeout_ms: Optional[int] = 60000,
) -> Dict[str, Any]:
    """Entity/relation extraction. Use when model's inference='extract-it' or for default Extract-It service.

    Args:
        text: Input text.
        labels: Target labels.
        threshold: Confidence threshold (0.0-1.0).
        model_id: Model ID, or None for default service.
        timeout_ms: Timeout in ms (default 60000).
        ctx: FastMCP request context.

    Returns:
        Extraction results.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
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
    ctx: Context,
    text: str,
    labels: List[str],
    threshold: Optional[float] = None,
    model_id: Optional[str] = None,
    timeout_ms: Optional[int] = 60000,
) -> Dict[str, Any]:
    """Text classification. Use when model's inference='comprehend-it' or for default Comprehend-It service.

    Args:
        text: Input text.
        labels: Classification labels.
        threshold: Confidence threshold (0.0-1.0).
        model_id: Model ID, or None for default service.
        timeout_ms: Timeout in ms (default 60000).
        ctx: FastMCP request context.

    Returns:
        Classification results.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
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


async def open_model_inference(
    ctx: Context,
    model_id: str,
    body: Dict[str, Any],
    timeout_ms: Optional[int] = 60000,
) -> Dict[str, Any]:
    """Run inference on a model. Use when get_model_details returns inference='open-models'.

    IMPORTANT: Call get_inference_details first to get the exact payload schema.
    Different models expect different fields (e.g., 'entities' for NER, 'labels' for classification).

    Args:
        model_id: Model ID.
        body: Request body matching the model's payload_schema from get_inference_details.
        timeout_ms: Timeout in ms (default 60000).
        ctx: FastMCP request context.

    Returns:
        Inference results.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    open_model = OpenModelAPI(configs=client.configs, model_id=model_id)
    open_model.base_headers.update(headers)
    response = await open_model.predict_async(body, timeout_ms=timeout_ms)
    return {"result": response}
