from typing import Any

from knowledgator_mcp.config import settings


def serialize(obj: Any) -> Any:
    """Serializes arbitrary Python objects into standard JSON-serializable structures.

    Handles Pydantic models (v1 and v2), primitives, containers, and falls back to str().
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    elif isinstance(obj, (dict, list, str, int, float, bool, type(None))):
        return obj
    return str(obj)


def build_frontend_url(resource_type: str, object_id: str) -> str:
    """Constructs a frontend URL for a specific resource object."""
    base_url = settings.platform_frontend_url.rstrip("/")
    if resource_type == "datasets":
        return f"{base_url}/datasets?id={object_id}"
    elif resource_type == "trainings":
        return f"{base_url}/fine-tuning?id={object_id}"
    elif resource_type == "data":
        return f"{base_url}/datasets?tab=rawData&raw_data_id={object_id}"
    elif resource_type == "models":
        return f"{base_url}/models/{object_id}"
    raise ValueError(f"Unknown resource type: {resource_type}")


def add_url_to_object(obj: dict, resource_type: str) -> dict:
    """Adds URL field to a single object response."""
    if isinstance(obj, dict) and "id" in obj:
        obj["url"] = build_frontend_url(resource_type, obj["id"])
    return obj


def add_urls_to_list_response(response: dict, resource_type: str, items_key: str) -> dict:
    """Adds URL field to each item in a list response."""
    items = response.get(items_key, [])
    for item in items:
        add_url_to_object(item, resource_type)
    return response

