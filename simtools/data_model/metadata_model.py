"""
Definition of metadata model for input to and output of simtools.
Follows CTAO top-level data model definition.

* data products submitted to SimPipe ('input')
* data products generated by SimPipe ('output')

"""

import logging
from importlib.resources import files

import jsonschema

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

    schema, schema_file = _load_schema(schema_file)

    try:
        jsonschema.validate(data, schema=schema)
    except jsonschema.exceptions.ValidationError:
        _logger.error(f"Failed using {schema}")
        raise
    _logger.debug(f"Succeeded using {schema_file}")


def get_default_metadata_dict(schema_file=None, observatory="CTA"):
    """
    Returns metadata schema with default values.
    Follows the CTA Top-Level Data Model.

    Parameters
    ----------
    schema_file: str
        Schema file (jsonschema format) used for validation
    observatory: str
        Observatory name

    Returns
    -------
    dict with reference schema


    """

    _logger.debug(f"Loading default schema from {schema_file}")
    schema, _ = _load_schema(schema_file)

    return _fill_defaults(schema["definitions"], observatory)


def _load_schema(schema_file=None):
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
        schema_file = files("simtools").joinpath("schemas/metadata.schema.yml")

    schema = gen.collect_data_from_yaml_or_dict(in_yaml=schema_file, in_dict=None)
    _logger.debug(f"Loading schema from {schema_file}")

    return schema, schema_file


def _resolve_references(yaml_data, observatory="CTA"):
    """
    Resolve references in yaml data and expand the received dictionary accordingly.

    Parameters
    ----------
    yaml_data: dict
        Dictionary with yaml data.
    observatory: str
        Observatory name

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
            if part in ("definitions", observatory):
                continue
            ref_data = ref_data.get(part, {})
        return ref_data

    def _resolve_references_recursive(data):
        if isinstance(data, dict):
            if "$ref" in data:
                ref = data["$ref"]
                resolved_data = expand_ref(ref)
                if isinstance(resolved_data, dict) and len(resolved_data) > 1:
                    return _resolve_references_recursive(resolved_data)
                return resolved_data
            return {k: _resolve_references_recursive(v) for k, v in data.items()}
        if isinstance(data, list):
            return [_resolve_references_recursive(item) for item in data]
        return data

    return _resolve_references_recursive(yaml_data)


def _fill_defaults(schema, observatory="CTA"):
    """
    Fill default values from json schema.

    Parameters
    ----------
    schema: dict
        Schema describing the input data.
    observatory: str
        Observatory name

    Returns
    -------
    dict
        Dictionary with default values.

    """

    defaults = {observatory: {}}

    schema = _resolve_references(schema[observatory])

    def _fill_defaults_recursive(subschema, current_dict):
        try:
            for prop, prop_schema in subschema["properties"].items():
                if "default" in prop_schema:
                    current_dict[prop] = prop_schema["default"]
                elif "type" in prop_schema:
                    if prop_schema["type"] == "object":
                        current_dict[prop] = {}
                        _fill_defaults_recursive(prop_schema, current_dict[prop])
                    elif prop_schema["type"] == "array":
                        current_dict[prop] = [{}]
                        if "items" in prop_schema and isinstance(prop_schema["items"], dict):
                            _fill_defaults_recursive(prop_schema["items"], current_dict[prop][0])
        except KeyError:
            msg = "Missing 'properties' key in schema."
            _logger.error(msg)
            raise

    _fill_defaults_recursive(schema, defaults[observatory])
    return defaults
