"""Pure, fail-closed source-slot contracts; no SDK, filesystem or model calls.

The caller supplies trusted Frame.model_json_schema() and len(actual_units).
Never infer this count from SOURCE-looking text in user/context content.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

SLOT_PROTOCOL = "aurel-source-slots-v1"
MAX_SOURCE_SLOTS = 6
MAX_VALIDATION_DEPTH = 64


class SlotSchemaError(ValueError):
    """An unsupported or inconsistent trusted schema; never a model response."""


class SlotValidationError(ValueError):
    """Safe diagnostic: codes/paths only, never response values or extra keys."""
    def __init__(self, code, path="$"):
        self.code, self.path = code, path
        super().__init__(code + " at " + path)


_ALLOWED = {"$defs", "$ref", "type", "properties", "required", "additionalProperties",
            "items", "anyOf", "enum", "const", "minLength", "title", "default", "description"}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _source_count(value):
    if type(value) is not int or not 1 <= value <= MAX_SOURCE_SLOTS:
        raise SlotSchemaError("trusted_source_count_must_be_integer_in_1_to_6")


def _strict_schema(node):
    """Strict structural subset; leave local references as references, not copies."""
    if type(node) is not dict or set(node) - _ALLOWED:
        raise SlotSchemaError("unsupported_schema_keyword_or_node")
    if "type" in node and (type(node["type"]) is not str or node["type"] not in _TYPES):
        raise SlotSchemaError("unsupported_schema_type")
    if not any(key in node for key in ("type", "$ref", "anyOf")):
        raise SlotSchemaError("unconstrained_schema_node")
    result = {key: copy.deepcopy(value) for key, value in node.items()
              if key not in {"title", "default", "minLength", "$defs", "properties", "items", "anyOf", "const"}}
    if "minLength" in node and (node.get("type") != "string" or type(node["minLength"]) is not int or node["minLength"] < 0):
        raise SlotSchemaError("invalid_min_length_constraint")
    if "$defs" in node:
        if type(node["$defs"]) is not dict:
            raise SlotSchemaError("invalid_definitions")
        result["$defs"] = {name: _strict_schema(value) for name, value in node["$defs"].items()}
    if "$ref" in node:
        if not isinstance(node["$ref"], str) or not node["$ref"].startswith("#/$defs/"):
            raise SlotSchemaError("only_local_definition_references_supported")
    if "const" in node:
        if "enum" in node:
            raise SlotSchemaError("ambiguous_const_and_enum")
        result["enum"] = [copy.deepcopy(node["const"])]
    if "enum" in result and (type(result["enum"]) is not list or not result["enum"]):
        raise SlotSchemaError("invalid_enum")
    if "anyOf" in node:
        if type(node["anyOf"]) is not list or not node["anyOf"]:
            raise SlotSchemaError("invalid_union")
        result["anyOf"] = [_strict_schema(branch) for branch in node["anyOf"]]
    if node.get("type") == "object":
        properties = node.get("properties")
        required = node.get("required", [])
        if (type(properties) is not dict or any(type(key) is not str for key in properties)
                or type(required) is not list or any(type(key) is not str for key in required)
                or len(set(required)) != len(required) or not set(required) <= set(properties)
                or type(node.get("additionalProperties", False)) is not bool):
            raise SlotSchemaError("unsupported_object_schema")
        result["properties"] = {key: _strict_schema(value) for key, value in properties.items()}
        result["required"] = list(properties)
        result["additionalProperties"] = False
    elif "properties" in node or "required" in node or "additionalProperties" in node:
        raise SlotSchemaError("object_keywords_without_object_type")
    if node.get("type") == "array":
        if type(node.get("items")) is not dict:
            raise SlotSchemaError("array_items_schema_required")
        result["items"] = _strict_schema(node["items"])
    elif "items" in node:
        raise SlotSchemaError("items_without_array_type")
    return result


def _resolve(root, reference):
    current = root
    for component in reference[2:].split("/"):
        key = component.replace("~1", "/").replace("~0", "~")
        if type(current) is not dict or key not in current:
            raise SlotSchemaError("unresolved_local_definition")
        current = current[key]
    if type(current) is not dict:
        raise SlotSchemaError("reference_does_not_resolve_to_schema")
    return current


def _check_references(node, root):
    if type(node) is dict:
        if "$ref" in node:
            _resolve(root, node["$ref"])
        for key in ("properties", "$defs"):
            for child in node.get(key, {}).values():
                _check_references(child, root)
        if "items" in node:
            _check_references(node["items"], root)
        for branch in node.get("anyOf", []):
            _check_references(branch, root)


def build_source_slot_response_format(frame_schema, source_count):
    """Compile trusted Frame fields into fixed slots with exact native indices.

    minLength is omitted only from the wire grammar for gateway compatibility;
    validate_source_slot_envelope still enforces the original Frame constraint.
    Unknown schema constraints fail before any model request, rather than vanish.
    """
    _source_count(source_count)
    frame = _strict_schema(copy.deepcopy(frame_schema))
    if frame.get("type") != "object" or "index" in frame["properties"]:
        raise SlotSchemaError("frame_must_be_object_without_reserved_index")
    definitions = frame.pop("$defs", None)
    properties = {}
    for index in range(source_count):
        slot = copy.deepcopy(frame)
        slot["properties"] = {"index": {"type": "integer", "enum": [index]}, **slot["properties"]}
        slot["required"] = list(slot["properties"])
        properties["record_" + str(index)] = slot
    schema = {"type": "object", "properties": properties,
              "required": list(properties), "additionalProperties": False}
    if definitions is not None:
        schema["$defs"] = definitions
    _check_references(schema, schema)
    try:
        digest = hashlib.sha256(json.dumps(schema, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:12]
    except (TypeError, ValueError):
        raise SlotSchemaError("schema_is_not_finite_json") from None
    return {"type": "json_schema", "json_schema": {
        "name": "aurel_slots_v1_n" + str(source_count) + "_" + digest,
        "strict": True, "schema": schema}}


def _same_literal(value, expected):
    # In particular, False must never masquerade as integer source index zero.
    if type(value) in (int, float) and type(expected) in (int, float):
        return value == expected
    return type(value) is type(expected) and value == expected


def _validate(value, schema, root, path, depth=0):
    if depth > MAX_VALIDATION_DEPTH:
        raise SlotValidationError("validation_depth_exceeded", path)
    if "$ref" in schema:
        _validate(value, _resolve(root, schema["$ref"]), root, path, depth + 1)
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                _validate(value, branch, root, path, depth + 1)
                break
            except SlotValidationError:
                continue
        else:
            raise SlotValidationError("no_union_branch_matches", path)
    kind = schema.get("type")
    valid_type = {"object": type(value) is dict, "array": type(value) is list,
                  "string": type(value) is str, "integer": type(value) is int,
                  "number": type(value) is int or (type(value) is float and math.isfinite(value)),
                  "boolean": type(value) is bool, "null": value is None}
    if kind is not None and not valid_type[kind]:
        raise SlotValidationError("wrong_value_type", path)
    if "const" in schema and not _same_literal(value, schema["const"]):
        raise SlotValidationError("constant_mismatch", path)
    if "enum" in schema and not any(_same_literal(value, item) for item in schema["enum"]):
        raise SlotValidationError("enum_mismatch", path)
    if kind == "object":
        properties = schema.get("properties", {})
        if not set(schema.get("required", [])) <= set(value):
            raise SlotValidationError("missing_required_field", path)
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise SlotValidationError("unexpected_field", path)
        for name, child in properties.items():
            if name in value:
                _validate(value[name], child, root, path + "." + name, depth + 1)
    elif kind == "array":
        for index, item in enumerate(value):
            _validate(item, schema["items"], root, path + "[" + str(index) + "]", depth + 1)
    elif kind == "string" and len(value) < schema.get("minLength", 0):
        raise SlotValidationError("min_length_violation", path)


def validate_source_slot_envelope(data, frame_schema, source_count):
    """Validate exact slots, shapes, indices and original Frame constraints.

    No filtering, default filling, SOURCE-index inference or response mutation.
    Object order is irrelevant; declared slot order alone determines records order.
    The caller must still run its original native record/Frame validation.
    """
    contract = build_source_slot_response_format(frame_schema, source_count)
    schema = contract["json_schema"]["schema"]
    _validate(data, schema, schema, "$")
    records = []
    for index in range(source_count):
        row = data["record_" + str(index)]
        frame_value = {key: row[key] for key in frame_schema["properties"]}
        _validate(frame_value, frame_schema, frame_schema, "$.record_" + str(index))
        records.append(copy.deepcopy(row))
    return {"records": records}


def decode_source_slot_response(raw_text, frame_schema, source_count):
    """Strict JSON decoding before mapping: duplicate source keys are ambiguity."""
    if type(raw_text) is not str:
        raise SlotValidationError("response_content_must_be_string")

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SlotValidationError("duplicate_json_key")
            result[key] = value
        return result

    def reject_constant(_):
        raise SlotValidationError("non_finite_json_number")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise SlotValidationError("non_finite_json_number")
        return number

    try:
        data = json.loads(raw_text, object_pairs_hook=object_pairs,
                          parse_constant=reject_constant, parse_float=finite_float)
    except SlotValidationError:
        raise
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise SlotValidationError("invalid_json") from None
    return validate_source_slot_envelope(data, frame_schema, source_count)
