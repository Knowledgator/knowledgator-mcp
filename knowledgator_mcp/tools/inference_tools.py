import json
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import Context

from knowledgator_mcp.utils import serialize
from knowledgator_mcp.client import get_client_context


async def instruct_inference(
    text: str,
    template: Union[str, Dict[str, Any]],
    output_type: Optional[str] = None,
    adapter_id: Optional[str] = None,
    timeout_ms: Optional[int] = 60000,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Instruction task. Use when model's inference='instruct-it' or for default Instruct-It service.

    Args:
        text: prompt/instruction.
        template: JSON schema template string or dictionary the output should follow (required).
        output_type: Optional hint for desired output format (json, string, markdown, etc.).
        adapter_id: Optional LoRA adapter ID for fine-tuned adapters (attaches KnowledgatorAPI-Model-ID header).
        timeout_ms: Optional request timeout in milliseconds (default: 60000 ms = 60s).
        ctx: Optional FastMCP request context.

    Note:
        For standalone full models (open-models), use 'open_model_inference' tool instead.

    Returns:
        Instruct result.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    template_str = json.dumps(template) if isinstance(template, dict) else template
    response = await client.instruct.it_async(
        text=text,
        template=template_str,
        output_type=output_type,
        adapter_id=adapter_id,
        server_url=server_url,
        http_headers=headers,
        timeout_ms=timeout_ms,
    )
    return serialize(response)


async def extract_inference(
    text: str,
    tasks: Optional[Dict[str, Any]] = None,
    labels: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    adapter_id: Optional[str] = None,
    model_id: Optional[str] = None,
    timeout_ms: Optional[int] = 60000,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Entity/relation extraction. Use when model's inference='extract-it' or for default Extract-It service.

    Args:
        text: Input text to extract entities or relations from.
        tasks: Optional dictionary of extraction tasks following the Unified Inference API
            (e.g., {"entities": {"type": "span", "labels": ["person", "location"], "threshold": 0.5}}).
        labels: Optional list of target entity/relation labels to extract (wrapped into a 'span' task if tasks not supplied).
        threshold: Optional confidence threshold float (0.0 to 1.0).
        adapter_id: Optional LoRA adapter ID for fine-tuned adapters (attaches KnowledgatorAPI-Model-ID header).
        model_id: Optional alias for adapter_id.
        timeout_ms: Optional request timeout in milliseconds (default: 60000 ms = 60s).
        ctx: Optional FastMCP request context.

    Note:
        For standalone full models (open-models), use 'open_model_inference' tool instead.

    Returns:
        Extraction results.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    eff_adapter_id = adapter_id or model_id
    if tasks is None:
        if labels is None:
            raise ValueError("Either 'tasks' or 'labels' must be provided for extract_inference.")
        task_def: Dict[str, Any] = {
            "type": "span",
            "labels": labels,
        }
        if threshold is not None:
            task_def["threshold"] = threshold
        tasks = {"entities": task_def}

    response = await client.extract.it_async(
        text=text,
        tasks=tasks,
        adapter_id=eff_adapter_id,
        server_url=server_url,
        http_headers=headers,
        timeout_ms=timeout_ms,
    )
    return serialize(response)


async def comprehend_inference(
    text: str,
    tasks: Optional[Dict[str, Any]] = None,
    labels: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    adapter_id: Optional[str] = None,
    model_id: Optional[str] = None,
    timeout_ms: Optional[int] = 60000,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Text classification. Use when model's inference='comprehend-it' or for default Comprehend-It service.

    Args:
        text: Input text.
        tasks: Optional dictionary of classification tasks following the Unified Inference API
            (e.g., {"topic": {"type": "classify", "labels": ["geography", "sport"], "threshold": 0.5}}).
        labels: Optional list of candidate classification labels (wrapped into a 'classify' task if tasks not supplied).
        threshold: Optional confidence threshold (0.0-1.0).
        adapter_id: Optional LoRA adapter ID for fine-tuned adapters (attaches KnowledgatorAPI-Model-ID header).
        model_id: Optional alias for adapter_id.
        timeout_ms: Optional request timeout in milliseconds (default: 60000 ms = 60s).
        ctx: Optional FastMCP request context.

    Note:
        For standalone full models (open-models), use 'open_model_inference' tool instead.

    Returns:
        Classification results.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    eff_adapter_id = adapter_id or model_id
    if tasks is None:
        if labels is None:
            raise ValueError("Either 'tasks' or 'labels' must be provided for comprehend_inference.")
        task_def: Dict[str, Any] = {
            "type": "classify",
            "labels": labels,
        }
        if threshold is not None:
            task_def["threshold"] = threshold
        tasks = {"topic": task_def}

    response = await client.comprehend.it_async(
        text=text,
        tasks=tasks,
        adapter_id=eff_adapter_id,
        server_url=server_url,
        http_headers=headers,
        timeout_ms=timeout_ms,
    )
    return serialize(response)


async def open_model_inference(
    model_id: str,
    data: Dict[str, Any],
    method: Optional[str] = None,
    timeout_ms: Optional[int] = 60000,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Executes prediction on a standalone full Open Model (POST /inference/{model_id}/).

    Args:
        model_id: The Open Model identifier (e.g. 'm-d9be7ede307943439a30c6f814026566').
        data: Inference input dictionary containing task payload (e.g. {'text': '...', 'labels': [...]}).
        method: Optional sub-method (e.g. 'relations' for /inference/{model_id}/{method}/).
        timeout_ms: Request timeout in milliseconds (default: 60000 ms = 60s).
        ctx: Optional FastMCP request context.

    Returns:
        Prediction outputs returned by the Open Model.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    model_api = client.model(model_id=model_id, method=method)
    if method:
        response = await model_api.call_async(method=method, body=data, timeout_ms=timeout_ms)
    else:
        response = await model_api.predict_async(body=data, timeout_ms=timeout_ms)
    return serialize(response)
