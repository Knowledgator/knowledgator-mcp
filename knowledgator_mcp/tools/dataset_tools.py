from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from mcp.server.fastmcp import Context

from knowledgator_mcp.client import get_client_context
from knowledgator_mcp.utils import serialize



async def generate_dataset_plan(
    prompt: str,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Generates a dataset creation plan based on a task prompt.

    Args:
        prompt: Detailed prompt describing the goal, target domain, and schema for the dataset.
        ctx: Optional FastMCP request context.

    Returns:
        Structured dataset generation plan with schema, sample structure, and execution steps.
    """
    client, server_url, headers = get_client_context(ctx=ctx)
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
    num_samples: int = 100,
    domains: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    entity_types: Optional[List[str]] = None,
    json_template: Optional[str] = None,
    ner_output_format: Optional[str] = "simple",
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Starts dataset generation / dataset creation pipeline on Knowledgator.

    Args:
        name: Name for the dataset (alphanumeric with hyphens/underscores).
        dataset_type: Service type ('ner', 'text-classification', 'classification', 'relation-extraction', 'instruct-it').
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
    return serialize(res)


async def list_datasets(
    offset: int = 0,
    limit: int = 20,
    name: Optional[str] = None,
    dataset_type: Optional[str] = None,
    ctx: Optional[Context] = None,
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
    return serialize(response)


async def get_dataset_details(
    dataset_id: str,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Gets details for a specific dataset by ID.

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
    return serialize(response)


def _sanitize_filename(name: str) -> str:
    """Sanitize arbitrary string into a filesystem-safe filename."""
    import re

    # Replace forbidden path characters, slashes, and control characters with underscore
    clean = re.sub(r'[\\/*?:"<>|\r\n\t\s]+', "_", name).strip("._ ")
    return clean or "dataset"


def _resolve_dest_path(
    dataset_name: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = "data",
) -> Path:
    """Safely resolve and sanitize target output file path (enforces .json)."""
    safe_default_name = _sanitize_filename(dataset_name)
    if not safe_default_name.lower().endswith(".json"):
        safe_default_name += ".json"

    if output_path:
        p = Path(output_path).expanduser()
        # If output_path points to an existing directory or ends with a slash:
        if p.is_dir() or output_path.endswith(("/", "\\")):
            dest_file = p / safe_default_name
        else:
            # Sanitize user/gemini provided filename while preserving directory structure
            safe_user_filename = _sanitize_filename(p.name)
            if not safe_user_filename.lower().endswith(".json"):
                safe_user_filename += ".json"
            dest_file = p.parent / safe_user_filename
    else:
        dest_dir = Path(output_dir or "data").expanduser()
        dest_file = dest_dir / safe_default_name

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    return dest_file.resolve()


def _validate_classification_dataset(data: Any) -> List[Dict[str, Any]]:
    """Validate that dataset data is a non-empty list of dicts with classification fields.

    Prevents uploading malformed or garbage data to the platform.
    """
    if not isinstance(data, list):
        raise ValueError(f"Dataset payload must be a JSON list of items, got {type(data).__name__}.")
    if len(data) == 0:
        raise ValueError("Dataset payload is empty (0 samples). Cannot upload an empty dataset.")

    text_keys = {"text", "input", "content", "sentence", "premise"}
    label_keys = {"label", "category", "target", "classes", "class", "true_labels", "all_labels"}

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Sample at index {idx} must be a dictionary, got {type(item).__name__}.")
        has_text = any(k in item and isinstance(item[k], str) and item[k].strip() for k in text_keys)
        if not has_text:
            raise ValueError(
                f"Sample at index {idx} is missing a valid text field (expected one of: {', '.join(sorted(text_keys))})."
            )
        has_label = any(k in item for k in label_keys)
        if not has_label:
            raise ValueError(
                f"Sample at index {idx} is missing a classification target/label (expected one of: {', '.join(sorted(label_keys))})."
            )

    return data


def _validate_dataset_payload(data: Any, dataset_type: str = "comprehend-it") -> List[Dict[str, Any]]:
    """Validate that dataset data is a non-empty list of dicts appropriate for its type."""
    if not isinstance(data, list):
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]
        else:
            raise ValueError(f"Dataset payload must be a JSON list of items, got {type(data).__name__}.")
    if len(data) == 0:
        raise ValueError("Dataset payload is empty (0 samples). Cannot upload an empty dataset.")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Sample at index {idx} must be a dictionary, got {type(item).__name__}.")

    if dataset_type == "comprehend-it":
        return _validate_classification_dataset(data)

    return data


def _sanitize_dataset_name(name: str, fallback: Optional[str] = None) -> str:
    """Sanitizes dataset name to satisfy OpenAPI pattern ^[a-zA-Z0-9- .()]+$ and length 2-64 without auto-adding version suffixes."""
    import re

    clean_name = name.strip() if name else ""
    if not clean_name and fallback:
        clean_name = fallback.strip()
    if not clean_name:
        clean_name = "Dataset"

    # Enforce OpenAPI pattern ^[a-zA-Z0-9- .()]+$
    clean_name = re.sub(r'[^a-zA-Z0-9\- .()]', '-', clean_name).strip()
    if len(clean_name) < 2:
        clean_name = f"{clean_name}-ds"[:64]
    elif len(clean_name) > 64:
        clean_name = clean_name[:64]

    return clean_name


async def download_dataset(
    dataset_id: str,
    timeout_ms: Optional[int] = None,
    ctx: Optional[Context] = None,
) -> Any:
    """Downloads an existing dataset by ID statelessly in memory.

    Streams chunks from the platform and passes the dataset samples/rows directly across
    the MCP wire to the caller (e.g. sandbox sync script) without writing any files to
    the MCP container disk.

    Args:
        dataset_id: Unique identifier of the dataset (e.g. 'd655eda9-ca67-4a86-a8bc-5b3911233c79').
        timeout_ms: Optional timeout override in milliseconds.
        ctx: Optional FastMCP request context.

    Returns:
        The raw dataset rows (typically List[Dict[str, Any]]).
    """
    import json
    import logging
    from knowledgator_mcp.config import settings

    logger = logging.getLogger("knowledgator_mcp.tools.dataset_tools")
    logger.info("Initiating stateless download for dataset: %s", dataset_id)

    client, server_url, headers = get_client_context(ctx=ctx)
    if client.datasets is None:
        raise RuntimeError("Datasets API is not enabled for the current client configuration.")

    # Resolve timeout: reuse settings.request_timeout_ms (env var KG_MCP_REQUEST_TIMEOUT_MS)
    effective_timeout_ms = timeout_ms or settings.request_timeout_ms

    try:
        # Prefer download_async with return_data=True for direct in-memory deserialization
        if hasattr(client.datasets, "download_async"):
            res = await client.datasets.download_async(
                dataset_id=dataset_id,
                return_data=True,
                server_url=server_url,
                timeout_ms=effective_timeout_ms,
                http_headers=headers,
            )
            if not res or (isinstance(res, (list, dict, bytes)) and len(res) == 0):
                raise RuntimeError(f"Downloaded content for dataset '{dataset_id}' is empty (0 bytes).")
            return res
        elif hasattr(client.datasets, "stream_download_async"):
            chunks: List[bytes] = []
            async for chunk in client.datasets.stream_download_async(
                dataset_id=dataset_id,
                server_url=server_url,
                timeout_ms=effective_timeout_ms,
                http_headers=headers,
            ):
                chunks.append(chunk)
            raw_bytes = b"".join(chunks)
        else:
            res = client.datasets.download(
                dataset_id=dataset_id,
                return_data=True,
                server_url=server_url,
                timeout_ms=effective_timeout_ms,
                http_headers=headers,
            )
            if not res or (isinstance(res, (list, dict, bytes)) and len(res) == 0):
                raise RuntimeError(f"Downloaded content for dataset '{dataset_id}' is empty (0 bytes).")
            return res

        if not raw_bytes or len(raw_bytes) == 0:
            raise RuntimeError(f"Downloaded content for dataset '{dataset_id}' is empty (0 bytes).")

        # Parse downloaded content as JSON
        try:
            return json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return raw_bytes.decode("utf-8", errors="replace")

    except Exception as exc:
        logger.error("Download failed for dataset %s: %s", dataset_id, exc)
        raise RuntimeError(
            f"Failed to download dataset {dataset_id}: {exc}. Please retry or run the sync script again."
        ) from exc


async def upload_dataset(
    name: str,
    data: Optional[Union[List[Any], Dict[str, Any], str]] = None,
    json_content: Optional[Union[str, List[Any], Dict[str, Any]]] = None,
    dataset_type: str = "classification",
    short_description: Optional[str] = None,
    parent_dataset_id: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Uploads a validated dataset to create a new dataset.

    Accepts dataset records in memory (data) or as a JSON string (json_content).
    Validates the dataset payload, sanitizes the dataset name, and streams directly to
    the backend via multipart form data without creating any files on the MCP container disk.

    Args:
        name: Name for the dataset (e.g. 'Customer Feedback Sentiment').
        data: Optional in-memory list or dict of sample data, or JSON string.
        json_content: Optional raw JSON string or parsed JSON object of dataset samples.
        dataset_type: Service type ('classification', 'ner', etc.).
        short_description: Optional short description (max 1024 characters).
        parent_dataset_id: Optional parent dataset UUID for version tracking.
        ctx: Optional FastMCP request context.

    Returns:
        Creation response with new dataset ID, name, sample count, and status.
    """
    import json
    import logging

    logger = logging.getLogger("knowledgator_mcp.tools.dataset_tools")

    # Resolve raw dataset data from either parameter
    raw_input = data if data is not None else json_content
    if raw_input is None:
        raise ValueError("Either 'data' or 'json_content' must be provided.")

    resolved_data: Any = None
    if isinstance(raw_input, str):
        try:
            resolved_data = json.loads(raw_input)
        except Exception as exc:
            raise ValueError(f"Invalid JSON provided in json_content: {exc}")
    else:
        resolved_data = raw_input

    # Normalize dataset type and validate schema
    type_map = {
        "classification": "comprehend-it",
        "text-classification": "comprehend-it",
        "comprehend-it": "comprehend-it",
        "ner": "extract-it",
        "named-entity-recognition": "extract-it",
        "extract-it": "extract-it",
        "relation-extraction": "extract-it",
        "text2json": "instruct-it",
        "instruct-it": "instruct-it",
    }
    normalized_type = type_map.get(dataset_type.lower().strip(), dataset_type)

    logger.info("Validating dataset payload with %s samples", len(resolved_data) if isinstance(resolved_data, list) else 0)
    validated_samples = _validate_dataset_payload(resolved_data, dataset_type=normalized_type)
    sample_count = len(validated_samples)

    clean_name = _sanitize_dataset_name(name)
    logger.info("Prepared clean dataset name: '%s'", clean_name)

    # Serialize to in-memory UTF-8 bytes (Stateless - no disk write on MCP)
    json_bytes = json.dumps(validated_samples, ensure_ascii=False, indent=2).encode("utf-8")
    size_bytes = len(json_bytes)

    # Call Knowledgator SDK client to create dataset + upload file
    client, server_url, headers = get_client_context(ctx=ctx)
    if client.datasets is None:
        raise RuntimeError("Datasets API is not enabled for the current client configuration.")

    create_kwargs: Dict[str, Any] = {
        "name": clean_name,
        "type": normalized_type,
        "format": "json",
        "short_description": short_description,
        "file": json_bytes,
        "server_url": server_url,
        "http_headers": headers,
    }

    if parent_dataset_id and parent_dataset_id != "none":
        create_kwargs["parent_dataset_id"] = parent_dataset_id

    try:
        if hasattr(client.datasets, "create_async"):
            creation_resp = await client.datasets.create_async(**create_kwargs)
        else:
            creation_resp = client.datasets.create(**create_kwargs)
    except Exception as exc:
        logger.error("Upload failed for dataset '%s': %s", clean_name, exc)
        raise RuntimeError(
            f"Failed to upload dataset '{clean_name}': {exc}. Please retry or run the sync script again."
        ) from exc

    serialized = serialize(creation_resp)
    new_id = serialized.get("id") if isinstance(serialized, dict) else str(creation_resp)
    dataset_status = serialized.get("status", "valid") if isinstance(serialized, dict) else "valid"

    logger.info("Successfully created dataset via SDK: %s", new_id)

    return {
        "status": "success",
        "id": new_id,
        "dataset_id": new_id,
        "dataset_name": clean_name,
        "sample_count": sample_count,
        "dataset_type": normalized_type,
        "format": "json",
        "size_bytes": size_bytes,
        "dataset_status": dataset_status,
        "short_description": short_description,
    }

