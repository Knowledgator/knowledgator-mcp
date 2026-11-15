from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import Context

from knowledgator_mcp.utils import serialize, add_url_to_object, add_urls_to_list_response
from knowledgator_mcp.client import get_client_context
from knowledgator_mcp.tools.reports import generate_training_metrics_report


async def start_training_job(
    model_id: str,
    dataset_id: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    type: str = "model",
    validation_dataset_id: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """CREATES and STARTS a new model fine-tuning training job.

    Call this tool when user wants to TRAIN or FINE-TUNE a model on a dataset.
    Do NOT call get_training_status first - that requires an existing training_id.

    Args:
        model_id: Base model identifier to fine-tune.
        dataset_id: Dataset identifier containing training samples.
        hyperparameters: Optional dictionary of training hyperparameters (seed, batch, learning_rate_mult, num_of_epochs).
        type: Training job type ('model', 'adapter', 'qlora', 'dpo', 'sft'). Defaults to 'model'.
        validation_dataset_id: Optional validation dataset identifier.
        ctx: Optional FastMCP request context.

    Returns:
        Training job details including training ID and status.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.finetuning is None:
        raise RuntimeError("Finetuning API is not enabled for the current client configuration.")
    response = client.finetuning.start_training(
        model_id=model_id,
        dataset_id=dataset_id,
        hyperparameters=hyperparameters,
        type=type,
        validation_dataset_id=validation_dataset_id,
        server_url=server_url,
        http_headers=headers,
    )
    return serialize(response)


async def get_training_status(
    training_id: str,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Gets status of an EXISTING training job.

    Use ONLY to check status of a training that was ALREADY STARTED via start_training_job.
    Will error if training_id doesn't exist - never call without a valid training_id.

    For detailed metrics (loss curve, accuracy, F1, precision, recall) and events,
    also use get_training_metrics_report.

    Status values: 'active' (in progress), 'completed' (success), 'failed', 'stopped'.

    Args:
        training_id: Unique training job identifier from a previously started job.
        ctx: Optional FastMCP request context.

    Returns:
        Training job details including status, progress, and model_id of the resulting model.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.finetuning is None:
        raise RuntimeError("Finetuning API is not enabled for the current client configuration.")
    response = client.finetuning.get_training(
        training_id=training_id,
        server_url=server_url,
        http_headers=headers,
    )
    result = serialize(response)
    return add_url_to_object(result, "trainings")


async def list_trainings(
    offset: int = 0,
    limit: int = 20,
    model_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    status: Optional[str] = None,
    type: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Lists training jobs with optional status/type/model/dataset filtering.

    Use this tool to inspect user's training jobs and find custom-trained models or adapters.
    Set status='completed' to see all user-trained models and adapters ready for inference.

    Args:
        offset: Pagination offset (default: 0).
        limit: Maximum number of jobs to return (default: 20).
        model_id: Optional filter by base model ID.
        dataset_id: Optional filter by dataset ID.
        status: Optional status filter ('active', 'completed', 'failed', 'stopped').
        type: Optional service type filter ('comprehend-it', 'extract-it', 'instruct-it').
        ctx: Optional FastMCP request context.

    Returns:
        List of training jobs with total count and pagination metadata.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.finetuning is None:
        raise RuntimeError("Finetuning API is not enabled for the current client configuration.")
    response = await client.finetuning.list_trainings_async(
        offset=offset,
        limit=limit,
        model_id=model_id,
        dataset_id=dataset_id,
        status=status,
        type=type,
        server_url=server_url,
        http_headers=headers,
    )
    result = serialize(response)
    return add_urls_to_list_response(result, "trainings", "trainings")


async def list_models(
    offset: int = 0,
    limit: int = 20,
    object_type: Optional[str] = None,
    trainable: Optional[str] = None,
    inference: Optional[str] = None,
    type: Optional[str] = None,
    name: Optional[str] = None,
    include_description: bool = False,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Lists available platform base models and public catalog adapters.

    Queries the Knowledgator platform foundation model catalog with pure server-side filtering.
    For user's custom-trained models and fine-tuned adapters, use list_trainings(status='completed').

    Args:
        offset: Pagination offset (default: 0).
        limit: Maximum number of models to return (default: 20).
        object_type: Filter by object type ('adapter' for LoRA adapters, 'model' for full models).
        trainable: Filter by trainable capability ('adapter' for models supporting adapter training, 'model' for full model training).
        inference: Filter by inference service ('open-models', 'comprehend-it', 'extract-it', 'instruct-it').
        type: Optional filter by architecture/model type (e.g. 'GLiClass', 'GLiFormer').
        name: Optional name filter substring.
        include_description: Whether to include full markdown description (default False to keep response compact). Full description is always available via get_model_details.
        ctx: Optional FastMCP request context.

    Returns:
        Dictionary with 'total', 'models' (or 'data') list of models, and pagination metadata.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.finetuning is None:
        raise RuntimeError("Finetuning API is not enabled for the current client configuration.")

    response = await client.finetuning.list_models_async(
        offset=offset,
        limit=limit,
        object_type=object_type,
        trainable=trainable,
        inference=inference,
        type=type,
        name=name,
        server_url=server_url,
        http_headers=headers,
    )
    result = serialize(response)
    items_key = "models" if isinstance(result, dict) and "models" in result else "data"
    if not include_description and isinstance(result, dict) and items_key in result:
        for item in result[items_key]:
            if isinstance(item, dict):
                item.pop("description", None)
    return add_urls_to_list_response(result, "models", items_key)


async def stop_training_job(
    training_id: str,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Stops a currently running fine-tuning training job.

    Args:
        training_id: Unique training job identifier to stop.
        ctx: Optional FastMCP request context.

    Returns:
        Confirmation response with updated status.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.finetuning is None:
        raise RuntimeError("Finetuning API is not enabled for the current client configuration.")
    response = client.finetuning.stop_training(
        training_id=training_id,
        server_url=server_url,
        http_headers=headers,
    )
    return serialize(response)


async def get_model_details(
    ctx: Context,
    model_id: str,
) -> Dict[str, Any]:
    """Gets model details to determine inference capabilities.

    IMPORTANT: Call this BEFORE running inference on a fine-tuned model to check
    what inference method to use:
    - If 'type' is 'extract-it': use extract_inference with adapter_id
    - If 'type' is 'comprehend-it': use comprehend_inference with adapter_id
    - If 'type' is 'instruct-it': use instruct_inference with adapter_id

    The 'training_model' and 'training_adapter' fields indicate what training
    types this model supports (for base models).

    Args:
        model_id: Model identifier (base model or fine-tuned model ID).
        ctx: Optional FastMCP request context.

    Returns:
        Model details including type, name, and inference capabilities.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.finetuning is None:
        raise RuntimeError("Finetuning API is not enabled for the current client configuration.")
    response = await client.finetuning.get_model_async(
        model_id=model_id,
        server_url=server_url,
        http_headers=headers,
    )
    result = serialize(response)
    return add_url_to_object(result, "models")


async def get_inference_details(
    ctx: Context,
    model_id: str,
) -> Dict[str, Any]:
    """Gets inference input schema for a model.

    IMPORTANT: Call this BEFORE running inference to get the exact parameters
    the model expects. Returns payload_schema and payload_example showing
    required fields (e.g., 'entities' for NER, 'labels' for classification).

    Args:
        model_id: Model identifier.
        ctx: FastMCP request context.

    Returns:
        Inference endpoints with payload_schema, payload_example, headers_schema.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    url = f"{server_url.rstrip('/')}/models/{model_id}/inference/"
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        response = await http_client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


async def get_training_metrics_report(
    ctx: Context,
    training_id: str,
) -> str:
    """Generates an HTML report with training metrics visualizations.

    Returns a self-contained HTML document with:
    - Loss curve over training steps (interactive line chart)
    - Evaluation metrics (accuracy, F1, precision, recall) as progress bars
    - Training events timeline

    Args:
        training_id: Unique training job identifier.
        ctx: FastMCP request context.

    Returns:
        Self-contained HTML document string.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.finetuning is None:
        raise RuntimeError("Finetuning API is not enabled for the current client configuration.")

    # Get training info
    training_info = serialize(client.finetuning.get_training(
        training_id=training_id,
        server_url=server_url,
        http_headers=headers,
    ))

    # Get loss metrics
    loss_data = []
    try:
        loss_response = client.finetuning.get_metrics(
            training_id=training_id,
            metric="loss",
            limit=500,
            server_url=server_url,
            http_headers=headers,
        )
        loss_data = serialize(loss_response).get("data", [])
    except Exception:
        pass

    # Get evaluation metrics from /trainings/{id}/evaluations/
    eval_metrics = {}
    try:
        eval_url = f"{server_url.rstrip('/')}/trainings/{training_id}/evaluations/"
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            eval_response = await http_client.get(eval_url, headers=headers)
            eval_response.raise_for_status()
            eval_metrics = eval_response.json()
    except Exception:
        pass

    # Get training events
    svg_content = generate_training_metrics_report(
        training_id=training_id,
        training_info=training_info,
        loss_metrics=loss_data,
        eval_metrics=eval_metrics,
    )

    return svg_content
