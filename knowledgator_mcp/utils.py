from typing import Any


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

