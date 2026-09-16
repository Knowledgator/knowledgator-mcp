from typing import Any, Dict, Optional

from knowledgator_mcp.utils import serialize
from knowledgator_mcp.client import get_client_context


async def start_training_job(
    model_id: str,
    dataset_id: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    type: str = "model",
    validation_dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Starts a model fine-tuning training job.

    Args:
        model_id: Base model identifier to fine-tune.
        dataset_id: Dataset identifier containing training samples.
        hyperparameters: Optional dictionary of training hyperparameters (seed, batch, learning_rate_mult, num_of_epochs).
        type: Training job type ('model', 'adapter', 'qlora', 'dpo', 'sft'). Defaults to 'model'.
        validation_dataset_id: Optional validation dataset identifier.

    Returns:
        Training job details including training ID and status.
    """
    client, server_url, headers = get_client_context()
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


async def get_training_status(training_id: str) -> Dict[str, Any]:
    """Gets details and status of a training job.

    Args:
        training_id: Unique training job identifier.

    Returns:
        Full details of the training job including status, metrics, and progress.
    """
    client, server_url, headers = get_client_context()
    if client.finetuning is None:
        raise RuntimeError("Finetuning API is not enabled for the current client configuration.")
    response = client.finetuning.get_training(
        training_id=training_id,
        server_url=server_url,
        http_headers=headers,
    )
    return serialize(response)


async def list_trainings(
    offset: int = 0,
    limit: int = 20,
    model_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Lists training jobs with optional status/model/dataset filtering.

    Args:
        offset: Pagination offset (default: 0).
        limit: Maximum number of jobs to return (default: 20).
        model_id: Optional filter by base model ID.
        dataset_id: Optional filter by dataset ID.
        status: Optional status filter ('pending', 'running', 'successful', 'failed', 'stopped').

    Returns:
        List of training jobs with total count and pagination metadata.
    """
    client, server_url, headers = get_client_context()
    if client.finetuning is None:
        raise RuntimeError("Finetuning API is not enabled for the current client configuration.")
    response = client.finetuning.list_trainings(
        offset=offset,
        limit=limit,
        model_id=model_id,
        dataset_id=dataset_id,
        status=status,
        server_url=server_url,
        http_headers=headers,
    )
    return serialize(response)


async def stop_training_job(training_id: str) -> Dict[str, Any]:
    """Stops a currently running fine-tuning training job.

    Args:
        training_id: Unique training job identifier to stop.

    Returns:
        Confirmation response with updated status.
    """
    client, server_url, headers = get_client_context()
    if client.finetuning is None:
        raise RuntimeError("Finetuning API is not enabled for the current client configuration.")
    response = client.finetuning.stop_training(
        training_id=training_id,
        server_url=server_url,
        http_headers=headers,
    )
    return serialize(response)
