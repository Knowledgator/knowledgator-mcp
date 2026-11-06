import logging
import traceback
import uuid
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import Context

from knowledgator_mcp.utils import serialize, add_url_to_object, add_urls_to_list_response
from knowledgator_mcp.client import get_client_context
from knowledgator_mcp.tools.reports import generate_dataset_analytics_report


async def generate_dataset_plan(
    prompt: str,
    ctx: Context,
) -> Dict[str, Any]:
    """REQUIRED FIRST STEP: Generates a dataset creation plan based on user's description.

    Call this tool FIRST before start_dataset_generation. The user MUST provide the prompt.
    Returns a plan that should be shown to user for confirmation before proceeding.

    Args:
        prompt: User's description of what dataset they want (REQUIRED - ask user if not provided).
        ctx: Optional FastMCP request context.

    Returns:
        Structured dataset generation plan for user to review and confirm.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.datasets is None:
        raise RuntimeError("Datasets API is not enabled for the current client configuration.")

    res = await client.datasets.generate_plan_async(
        prompt=prompt,
        server_url=server_url,
        http_headers=headers,
    )
    result = serialize(res)
    result["plan_id"] = str(uuid.uuid4())
    return result


async def start_dataset_generation(
    ctx: Context,
    name: str,
    dataset_type: str,
    plan_id: Optional[str] = None,
    prompt: Optional[str] = None,
    description: Optional[str] = None,
    num_samples: int = 100,
    domains: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    entity_types: Optional[List[str]] = None,
    json_template: Optional[str] = None,
    ner_output_format: Optional[str] = "simple",
) -> Dict[str, Any]:
    """SECOND STEP: Starts dataset generation AFTER user confirms the plan from generate_dataset_plan.

    WORKFLOW: generate_dataset_plan -> user confirms -> start_dataset_generation
    Use parameters from the confirmed plan. Do NOT call without user confirmation.

    Args:
        name: Name for the dataset (alphanumeric with hyphens/underscores).
        dataset_type: Service type ('ner', 'text-classification', 'classification', 'relation-extraction', 'instruct-it').
        plan_id: Plan ID from generate_dataset_plan response.
        prompt: Task prompt describing the data to generate. If omitted, uses description or name.
        description: Optional description of the dataset purpose and prompt specifications.
        num_samples: Number of samples to generate (default: 100).
        domains: Optional list of domain tags (e.g. ['e-commerce', 'finance']).
        classes: Optional list of classification target categories (for classification / text-classification).
        entity_types: Optional list of entity tag strings (for NER / extract-it).
        json_template: Optional JSON schema template string (for text2json / instruct-it).
        ner_output_format: Output format for NER ('simple', etc.).
        ctx: Optional FastMCP request context.

    Returns:
        Creation response with dataset ID and status.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
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
    result = serialize(res)
    result["plan_id"] = plan_id
    return result


async def list_datasets(
    ctx: Context,
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
        ctx: Optional FastMCP request context.

    Returns:
        List of datasets with total count and pagination info.
    """
    try:
        client, server_url, headers = get_client_context(ctx=ctx)
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
        result = serialize(response)
        result = add_urls_to_list_response(result, "datasets", "datasets")
        return result
    except Exception as e:
        logging.exception(e)
        raise


async def get_dataset_details(
    dataset_id: str,
    ctx: Context,
) -> Dict[str, Any]:
    """Gets metadata for a specific dataset by ID.

    For detailed statistics (quality profile, label distribution, token stats),
    also use get_dataset_analytics_report.

    Args:
        dataset_id: Unique identifier of the dataset.
        ctx: Optional FastMCP request context.

    Returns:
        Dataset metadata, file counts, and processing status.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.datasets is None:
        raise RuntimeError("Datasets API is not enabled for the current client configuration.")
    response = client.datasets.get(
        dataset_id=dataset_id,
        server_url=server_url,
        http_headers=headers,
    )
    result = serialize(response)
    return add_url_to_object(result, "datasets")


async def get_dataset_analytics_report(
    dataset_id: str,
    ctx: Context,
) -> str:
    """Generates an HTML report with dataset analytics visualizations.

    Returns a self-contained HTML document with:
    - Quality profile radar chart (balance, quality, diversity, complexity, consistency)
    - Label distribution bar chart with balance indicator
    - Text statistics table (tokens, labels - min/max/mean/median/std)
    - Token count distribution histogram
    - Classification statistics (labels per text)

    Args:
        dataset_id: Unique identifier of the dataset.
        ctx: FastMCP request context.

    Returns:
        Self-contained HTML document string.
    """
    client, server_url, headers = get_client_context(ctx=ctx)

    # Fetch statistics directly from API (not in SDK yet)
    stats_url = f"{server_url.rstrip('/')}/datasets/{dataset_id}/statistics/"

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        response = await http_client.get(stats_url, headers=headers)
        response.raise_for_status()
        statistics = response.json()

    svg_content = generate_dataset_analytics_report(
        dataset_id=dataset_id,
        statistics=statistics,
    )

    return svg_content


async def list_data(
    ctx: Context,
    offset: int = 0,
    limit: int = 100,
    name: Optional[str] = None,
    data_type: Optional[str] = None,
    data_format: Optional[str] = None,
) -> dict[str, Any]:
    """Lists raw data on the Knowledgator platform with optional filtering.

    Args:
        offset: Pagination offset (default: 0).
        limit: Maximum number of data items to return (default: 100).
        name: Optional filter by data name.
        data_type: Optional filter by service type.
        data_format: Optional filter by data format.
        ctx: FastMCP request context.

    Returns:
        List of raw data with total count and pagination info.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.datasets is None:
        raise RuntimeError("Datasets API is not enabled for the current client configuration.")
    response = client.datasets.list_data(
        offset=offset,
        limit=limit,
        name=name,
        type=data_type,
        format=data_format,
        server_url=server_url,
        http_headers=headers,
    )
    result = serialize(response)
    return add_urls_to_list_response(result, "data", "data")
