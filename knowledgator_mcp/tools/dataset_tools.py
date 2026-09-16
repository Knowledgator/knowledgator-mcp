from typing import Any, Dict, List, Optional

from knowledgator_mcp.utils import serialize
from knowledgator_mcp.client import get_client_context


async def generate_dataset_plan(
    prompt: str,
    dataset_type: str = "text-classification",
    target_format: str = "jsonl",
) -> Dict[str, Any]:
    """Generates a dataset creation plan based on a task prompt.

    Args:
        prompt: Detailed prompt describing the goal, target domain, and schema for the dataset.
        dataset_type: Type of task (e.g., 'text-classification', 'ner', 'relation-extraction').
        target_format: Desired dataset format ('jsonl', 'csv', 'parquet', 'json').

    Returns:
        Structured dataset generation plan with schema, sample structure, and execution steps.
    """
    client, server_url, headers = get_client_context()
    if client.datasets is None:
        raise RuntimeError("Datasets API is not enabled for the current client configuration.")

    res = await client.datasets.generate_plan_async(
        prompt=prompt,
        server_url=server_url,
        http_headers=headers,
    )
    return serialize(res)


async def start_dataset_generation(
    name: str,
    dataset_type: str,
    prompt: Optional[str] = None,
    description: Optional[str] = None,
    format: str = "jsonl",
    num_samples: int = 100,
    domains: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    entity_types: Optional[List[str]] = None,
    json_template: Optional[str] = None,
    ner_output_format: Optional[str] = "simple",
) -> Dict[str, Any]:
    """Starts dataset generation / dataset creation pipeline on Knowledgator.

    Args:
        name: Name for the dataset (alphanumeric with hyphens/underscores).
        dataset_type: Service type ('ner', 'text-classification', 'classification', 'relation-extraction', 'instruct-it').
        prompt: Task prompt describing the data to generate. If omitted, uses description or name.
        description: Optional description of the dataset purpose and prompt specifications.
        format: Dataset format ('jsonl', 'csv', 'parquet', 'json').
        num_samples: Number of samples to generate (default: 100).
        domains: Optional list of domain tags (e.g. ['e-commerce', 'finance']).
        classes: Optional list of classification target categories (for classification / text-classification).
        entity_types: Optional list of entity tag strings (for NER / extract-it).
        json_template: Optional JSON schema template string (for text2json / instruct-it).
        ner_output_format: Output format for NER ('simple', etc.).

    Returns:
        Creation response with dataset ID and status.
    """
    client, server_url, headers = get_client_context()
    if client.datasets is None:
        raise RuntimeError("Datasets API is not enabled for the current client configuration.")

    # Normalize dataset type for generate endpoint if needed
    gen_type = dataset_type
    if dataset_type == "text-classification":
        gen_type = "classification"

    res = await client.datasets.generate_async(
        name=name,
        prompt=prompt or description or name,
        type=gen_type,
        num_samples=num_samples,
        domains=domains,
        classes=classes,
        entity_types=entity_types,
        json_template=json_template,
        ner_output_format=ner_output_format,
        server_url=server_url,
        http_headers=headers,
    )
    return serialize(res)


async def list_datasets(
    offset: int = 0,
    limit: int = 20,
    name: Optional[str] = None,
    dataset_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Lists datasets on the Knowledgator platform with optional filtering.

    Args:
        offset: Pagination offset (default: 0).
        limit: Maximum number of datasets to return (default: 20).
        name: Optional filter by dataset name.
        dataset_type: Optional filter by service type ('ner', 'text-classification', 'relation-extraction').

    Returns:
        List of datasets with total count and pagination info.
    """
    client, server_url, headers = get_client_context()
    if client.datasets is None:
        raise RuntimeError("Datasets API is not enabled for the current client configuration.")
    response = client.datasets.list(
        offset=offset,
        limit=limit,
        name=name,
        type=dataset_type,
        server_url=server_url,
        http_headers=headers,
    )
    return serialize(response)


async def get_dataset_details(dataset_id: str) -> Dict[str, Any]:
    """Gets details for a specific dataset by ID.

    Args:
        dataset_id: Unique identifier of the dataset.

    Returns:
        Dataset metadata, file counts, and processing status.
    """
    client, server_url, headers = get_client_context()
    if client.datasets is None:
        raise RuntimeError("Datasets API is not enabled for the current client configuration.")
    response = client.datasets.get(
        dataset_id=dataset_id,
        server_url=server_url,
        http_headers=headers,
    )
    return serialize(response)
