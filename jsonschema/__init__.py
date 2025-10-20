"""Lightweight JSON Schema Draft-07 validator used for offline testing."""

from __future__ import annotations

import datetime as _dt
import uuid as _uuid
from typing import Any, Iterable

from .exceptions import ValidationError


class _ExceptionsModule:
    ValidationError = ValidationError


exceptions = _ExceptionsModule()


def validate(instance: Any, schema: dict[str, Any]) -> None:
    """Validate *instance* against *schema*.

    Supports the JSON Schema features used by the NeuroLens Phase 0 artifacts.
    Raises :class:`ValidationError` when validation fails.
    """

    def _fail(message: str, path: list[Any]) -> None:
        raise ValidationError(message, path=path.copy())

    def _check_type(expected: str | Iterable[str], value: Any, path: list[Any]) -> None:
        if isinstance(expected, str):
            expected_types = {expected}
        else:
            expected_types = set(expected)
        for typ in list(expected_types):
            if typ == "null" and value is None:
                return
            if typ == "boolean" and isinstance(value, bool):
                return
            if typ == "integer" and isinstance(value, int) and not isinstance(value, bool):
                return
            if typ == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
                return
            if typ == "string" and isinstance(value, str):
                return
            if typ == "array" and isinstance(value, list):
                return
            if typ == "object" and isinstance(value, dict):
                return
        _fail(f"Expected type {expected_types}, got {type(value).__name__}", path)

    def _check_enum(enum_values: list[Any], value: Any, path: list[Any]) -> None:
        if value not in enum_values:
            _fail(f"Value {value!r} not in enum {enum_values}", path)

    def _check_number_constraints(subschema: dict[str, Any], value: Any, path: list[Any]) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        if "minimum" in subschema and value < subschema["minimum"]:
            _fail(f"Value {value} < minimum {subschema['minimum']}", path)
        if "maximum" in subschema and value > subschema["maximum"]:
            _fail(f"Value {value} > maximum {subschema['maximum']}", path)
        if subschema.get("exclusiveMinimum") is not None:
            exclusive_min = subschema["exclusiveMinimum"]
            if value <= exclusive_min:
                _fail(f"Value {value} <= exclusiveMinimum {exclusive_min}", path)
        if subschema.get("exclusiveMaximum") is not None:
            exclusive_max = subschema["exclusiveMaximum"]
            if value >= exclusive_max:
                _fail(f"Value {value} >= exclusiveMaximum {exclusive_max}", path)

    def _check_string_constraints(subschema: dict[str, Any], value: Any, path: list[Any]) -> None:
        if not isinstance(value, str):
            return
        fmt = subschema.get("format")
        if fmt == "uuid":
            try:
                _uuid.UUID(value)
            except Exception as exc:
                _fail(f"Invalid uuid string: {exc}", path)
        elif fmt == "date-time":
            try:
                if value.endswith("Z"):
                    _dt.datetime.fromisoformat(value[:-1] + "+00:00")
                else:
                    _dt.datetime.fromisoformat(value)
            except Exception as exc:
                _fail(f"Invalid date-time string: {exc}", path)

    def _validate(instance: Any, schema: dict[str, Any], path: list[Any]) -> None:
        if not isinstance(schema, dict):
            _fail("Schema must be an object", path)
        if "type" in schema:
            _check_type(schema["type"], instance, path)
        if "enum" in schema:
            _check_enum(schema["enum"], instance, path)
        if isinstance(instance, (int, float)) and not isinstance(instance, bool):
            _check_number_constraints(schema, instance, path)
        if isinstance(instance, str):
            _check_string_constraints(schema, instance, path)
        if schema.get("type") == "array" or isinstance(instance, list):
            if "minItems" in schema and len(instance) < schema["minItems"]:
                _fail(f"Array shorter than minItems {schema['minItems']}", path)
            items_schema = schema.get("items")
            if items_schema is not None:
                for index, item in enumerate(instance):
                    _validate(item, items_schema, path + [index])
        if schema.get("type") == "object" or isinstance(instance, dict):
            if not isinstance(instance, dict):
                _fail("Expected object", path)
            required = schema.get("required", [])
            for key in required:
                if key not in instance:
                    _fail(f"Missing required property '{key}'", path)
            properties = schema.get("properties", {})
            for key, value in instance.items():
                if key in properties:
                    _validate(value, properties[key], path + [key])
                else:
                    additional = schema.get("additionalProperties", True)
                    if additional is False:
                        _fail(f"Unexpected property '{key}'", path + [key])
                    elif isinstance(additional, dict):
                        _validate(value, additional, path + [key])
        # Recurse into subschemas referenced in allOf/anyOf? not needed here.

    _validate(instance, schema, [])


__all__ = ["validate", "exceptions", "ValidationError"]
