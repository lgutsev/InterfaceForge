"""Dependency-free checker for the JSON-Schema subset used by ``schemas/``.

InterfaceForge has no ``jsonschema`` runtime dependency, so the assignment and
audit schemas are enforced by this small interpreter. It supports exactly the
keywords those schemas use: ``type``, ``required``, ``properties``,
``additionalProperties``, ``items``, ``enum``, ``const``, ``minLength``,
``minItems``, ``pattern``, ``minimum``, ``exclusiveMinimum``, ``allOf``,
``anyOf``, ``oneOf``, ``not``, ``if``/``then`` and local ``$ref``
(``#/$defs/...``). Any other keyword raises, so a schema cannot silently ask for
a check that is not performed.

The canonical schemas live in the repository's ``schemas/`` directory; the
copies under ``interfaceforge/schemas/`` are package data, and a test keeps the
two byte-identical.
"""

from __future__ import annotations

import json
import re
from functools import cache
from importlib import resources
from typing import Any

_SUPPORTED = {
    "$schema",
    "$id",
    "$defs",
    "title",
    "description",
    "type",
    "required",
    "properties",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "minLength",
    "minItems",
    "pattern",
    "minimum",
    "exclusiveMinimum",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "$ref",
}

_TYPES: dict[str, Any] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


@cache
def load_schema(name: str) -> dict[str, Any]:
    """Load a bundled schema by file name, e.g. ``assignment.schema.json``."""

    text = resources.files("interfaceforge").joinpath("schemas", name).read_text(encoding="utf-8")
    return json.loads(text)


def _pointer(path: tuple[Any, ...]) -> str:
    return "/" + "/".join(str(part) for part in path) if path else "/"


def _resolve(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Only local $ref is supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _errors(schema: dict[str, Any], value: Any, root: dict[str, Any], path: tuple[Any, ...]) -> list[str]:
    unknown = set(schema) - _SUPPORTED
    if unknown:
        raise ValueError(f"schema keyword(s) not supported by schema_check: {sorted(unknown)}")
    where = _pointer(path)
    out: list[str] = []

    if "$ref" in schema:
        out += _errors(_resolve(root, schema["$ref"]), value, root, path)

    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[name](value) for name in allowed):
            return out + [f"{where}: expected {' or '.join(allowed)}, got {type(value).__name__}"]

    if "const" in schema and value != schema["const"]:
        out.append(f"{where}: must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{where}: {value!r} is not one of {schema['enum']}")

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            out.append(f"{where}: string shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            out.append(f"{where}: {value!r} does not match {schema['pattern']}")

    if _TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{where}: {value} < minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            out.append(f"{where}: {value} <= exclusiveMinimum {schema['exclusiveMinimum']}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                out.append(f"{where}: missing required field {key!r}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                out += _errors(properties[key], item, root, path + (key,))
            else:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    out.append(f"{where}: unexpected field {key!r}")
                elif isinstance(extra, dict):
                    out += _errors(extra, item, root, path + (key,))

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            out.append(f"{where}: fewer than {schema['minItems']} item(s)")
        if "items" in schema:
            for index, item in enumerate(value):
                out += _errors(schema["items"], item, root, path + (index,))

    for sub in schema.get("allOf", []):
        out += _errors(sub, value, root, path)
    if "anyOf" in schema and not any(not _errors(sub, value, root, path) for sub in schema["anyOf"]):
        out.append(f"{where}: matches none of the anyOf alternatives")
    if "oneOf" in schema:
        results = [_errors(sub, value, root, path) for sub in schema["oneOf"]]
        matches = sum(1 for result in results if not result)
        if matches != 1:
            closest = min(results, key=len)
            detail = "; ".join(closest[:5]) if matches == 0 else "matches more than one alternative"
            out.append(f"{where}: must match exactly one oneOf alternative ({detail})")
    if "not" in schema and not _errors(schema["not"], value, root, path):
        out.append(f"{where}: matches a forbidden pattern ({json.dumps(schema['not'], sort_keys=True)})")
    if "if" in schema and not _errors(schema["if"], value, root, path) and "then" in schema:
        out += _errors(schema["then"], value, root, path)
    return out


def schema_errors(value: Any, schema: dict[str, Any] | str) -> list[str]:
    """Return human-readable violations of ``schema`` (a dict or bundled file name)."""

    resolved = load_schema(schema) if isinstance(schema, str) else schema
    return _errors(resolved, value, resolved, ())
