from pydantic import BaseModel

from knowledgator_mcp.utils import serialize


class DummyPydanticModel(BaseModel):
    name: str
    value: int


class CustomLegacyModel:
    def dict(self):
        return {"legacy": True}


class CustomClass:
    def __str__(self):
        return "custom_object"


def test_serialize_primitives():
    assert serialize(42) == 42
    assert serialize(3.14) == 3.14
    assert serialize("hello") == "hello"
    assert serialize(True) == True
    assert serialize(None) is None
    assert serialize([1, 2, 3]) == [1, 2, 3]
    assert serialize({"a": 1}) == {"a": 1}


def test_serialize_pydantic_model():
    model = DummyPydanticModel(name="test", value=100)
    assert serialize(model) == {"name": "test", "value": 100}


def test_serialize_legacy_dict_model():
    model = CustomLegacyModel()
    assert serialize(model) == {"legacy": True}


def test_serialize_fallback_str():
    obj = CustomClass()
    assert serialize(obj) == "custom_object"
