"""
Definition of metadata model for input to and output of simtools.
Follows CTAO top-level data model definition.

* data products submitted to SimPipe ('input')
* data products generated by SimPipe ('output')

"""

import logging
from importlib.resources import files

import jsonschema
import yaml

import simtools.utils.general as gen

_logger = logging.getLogger(__name__)


def validate_schema(data, schema_file):
    """
    Validate dictionary against schema.

    Parameters
    ----------
    data
        dictionary to be validated
    schema_file (dict)
        schema used for validation

    Raises
    ------
    FileNotFoundError
        if input file is not found

    """

    schema = load_schema(schema_file)

    try:
        jsonschema.validate(data, schema=schema)
    except jsonschema.exceptions.ValidationError:
        _logger.error(f"Schema validation failed using {schema}")
        raise
    _logger.info("Schema validation successful")


def load_schema(schema_file=None):
    """
    Load parameter schema from file from simpipe metadata schema.

    Returns
    -------
    schema_file (dict)
        schema

    Raises
    ------
    FileNotFoundError
        if schema file is not found

    """

    if schema_file is None:
        with files("simtools").joinpath("schemas/metadata.schema.yml").open(
            "r", encoding="utf-8"
        ) as file:
            schema = yaml.safe_load(file)
        _logger.info("Loading schema from simtools package.")
    else:
        schema = gen.collect_data_from_yaml_or_dict(in_yaml=schema_file, in_dict=None)
        _logger.info(f"Loading schema from {schema_file}")

    return schema


def top_level_reference_schema(schema_file=None):
    """
    Reference schema following the CTA Top-Level Data Model.

    This metadata schema is used for simtools data products.

    Parameters
    ----------
    schema_file: str
        Schema file (jsonschema format) used for validation

    Returns
    -------
    dict with reference schema


    """

    schema = load_schema(schema_file)
    return fill_defaults(schema["definitions"]["CTA"])


def resolve_references(yaml_data):
    """
    Resolve references in yaml data and expand the received dictionary accordingly.

    Parameters
    ----------
    yaml_data: dict
        Dictionary with yaml data.

    Returns
    -------
    dict
        Dictionary with resolved references.

    """

    def expand_ref(ref):
        ref_path = ref.lstrip("#/")
        parts = ref_path.split("/")
        ref_data = yaml_data
        for part in parts:
            if part in ("definitions", "CTA"):
                continue
            ref_data = ref_data.get(part, {})
        return ref_data

    def resolve_references_recursive(data):
        if isinstance(data, dict):
            if "$ref" in data:
                ref = data["$ref"]
                resolved_data = expand_ref(ref)
                if isinstance(resolved_data, dict) and len(resolved_data) > 1:
                    return resolve_references_recursive(resolved_data)
                return resolved_data
            return {k: resolve_references_recursive(v) for k, v in data.items()}
        if isinstance(data, list):
            return [resolve_references_recursive(item) for item in data]
        return data

    return resolve_references_recursive(yaml_data)


def fill_defaults(schema):
    """
    Fill default values from json schema.

    Parameters
    ----------
    schema: dict
        Schema describing the input data.

    Returns
    -------
    dict
        Dictionary with default values.

    """

    defaults = {"CTA": {}}

    schema = resolve_references(schema)

    def fill_defaults_recursive(subschema, current_dict):
        if "properties" in subschema:
            for prop, prop_schema in subschema["properties"].items():
                if "default" in prop_schema:
                    current_dict[prop] = prop_schema["default"]
                elif "type" in prop_schema:
                    if prop_schema["type"] == "object":
                        current_dict[prop] = {}
                        fill_defaults_recursive(prop_schema, current_dict[prop])
                    elif prop_schema["type"] == "array":
                        current_dict[prop] = [{}]
                        if "items" in prop_schema and isinstance(prop_schema["items"], dict):
                            fill_defaults_recursive(prop_schema["items"], current_dict[prop][0])

    fill_defaults_recursive(schema, defaults["CTA"])
    return defaults
