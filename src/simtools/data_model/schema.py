"""Module providing functionality to read and validate dictionaries using schema."""

import logging
from pathlib import Path

import jsonschema

import simtools.utils.general as gen
from simtools.constants import (
    METADATA_JSON_SCHEMA,
    MODEL_PARAMETER_METASCHEMA,
    MODEL_PARAMETER_SCHEMA_PATH,
    SCHEMA_PATH,
)
from simtools.data_model import format_checkers
from simtools.utils import names

_logger = logging.getLogger(__name__)


def get_get_model_parameter_schema_files(schema_directory=MODEL_PARAMETER_SCHEMA_PATH):
    """
    Return list of parameters and schema files located in schema file directory.

    Returns
    -------
    list
        List of parameters found in schema file directory.
    list
        List of schema files found in schema file directory.

    """
    schema_files = sorted(Path(schema_directory).rglob("*.schema.yml"))
    if not schema_files:
        raise FileNotFoundError(f"No schema files found in {schema_directory}")
    parameters = []
    for schema_file in schema_files:
        schema_dict = gen.collect_data_from_file(file_name=schema_file)
        parameters.append(schema_dict.get("name"))
    return parameters, schema_files


def get_model_parameter_schema_file(parameter):
    """
    Return schema file path for a given model parameter.

    Parameters
    ----------
    parameter: str
        Model parameter name.

    Returns
    -------
    Path
        Schema file path.

    """
    schema_file = MODEL_PARAMETER_SCHEMA_PATH / f"{parameter}.schema.yml"
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    return schema_file


def get_model_parameter_schema_version(schema_version=None):
    """
    Validate  and return schema versions.

    If no schema_version is given, the most recent version is provided.

    Parameters
    ----------
    schema_version: str
        Schema version.

    Returns
    -------
    str
        Schema version.

    """
    schemas = gen.collect_data_from_file(MODEL_PARAMETER_METASCHEMA)

    if schema_version is None and schemas:
        return schemas[0].get("version")

    if any(schema.get("version") == schema_version for schema in schemas):
        return schema_version

    raise ValueError(f"Schema version {schema_version} not found in {MODEL_PARAMETER_METASCHEMA}.")


def validate_dict_using_schema(data, schema_file=None, json_schema=None):
    """
    Validate a data dictionary against a schema.

    Parameters
    ----------
    data
        dictionary to be validated
    schema_file (dict)
        schema used for validation

    Raises
    ------
    jsonschema.exceptions.ValidationError
        if validation fails

    """
    if json_schema is None and schema_file is None:
        _logger.warning(f"No schema provided for validation of {data}")
        return
    if json_schema is None:
        json_schema = load_schema(
            schema_file,
            data.get("schema_version", "0.1.0"),  # default version to ensure backward compatibility
        )

    try:
        jsonschema.validate(data, schema=json_schema, format_checker=format_checkers.format_checker)
    except jsonschema.exceptions.ValidationError as exc:
        _logger.error(f"Validation failed using schema: {json_schema} for data: {data}")
        raise exc
    if data.get("meta_schema_url") and not gen.url_exists(data["meta_schema_url"]):
        raise FileNotFoundError(f"Meta schema URL does not exist: {data['meta_schema_url']}")

    _logger.debug(f"Successful validation of data using schema ({json_schema.get('name')})")


def load_schema(schema_file=None, schema_version=None):
    """
    Load parameter schema from file.

    Parameters
    ----------
    schema_file: str
        Path to schema file.
    schema_version: str
        Schema version.

    Returns
    -------
    schema: dict
        Schema dictionary.

    Raises
    ------
    FileNotFoundError
        if schema file is not found

    """
    schema_file = schema_file or METADATA_JSON_SCHEMA

    for path in (schema_file, SCHEMA_PATH / schema_file):
        try:
            schema = gen.collect_data_from_file(file_name=path)
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    if isinstance(schema, list):  # schema file with several schemas defined
        if schema_version is None:
            raise ValueError(f"Schema version not given in {schema_file}.")
        schema = next((doc for doc in schema if doc.get("version") == schema_version), None)
        if schema is None:
            raise ValueError(f"Schema version {schema_version} not found in {schema_file}.")
    elif schema_version is not None and schema_version != schema.get("version"):
        _logger.warning(f"Schema version {schema_version} does not match {schema.get('version')}")

    _logger.debug(f"Loading schema from {schema_file}")
    _add_array_elements("InstrumentTypeElement", schema)

    return schema


def _add_array_elements(key, schema):
    """
    Add list of array elements to schema.

    Avoids having to list all array elements in multiple schema.
    Assumes an element [key]['enum'] is a list of elements.

    Parameters
    ----------
    key: str
        Key in schema dictionary
    schema: dict
        Schema dictionary

    Returns
    -------
    dict
        Schema dictionary with added array elements.

    """
    _list_of_array_elements = sorted(names.array_elements().keys())

    def recursive_search(sub_schema, key):
        if key in sub_schema:
            if "enum" in sub_schema[key] and isinstance(sub_schema[key]["enum"], list):
                sub_schema[key]["enum"] = list(
                    set(sub_schema[key]["enum"] + _list_of_array_elements)
                )
            else:
                sub_schema[key]["enum"] = _list_of_array_elements
        else:
            for _, v in sub_schema.items():
                if isinstance(v, dict):
                    recursive_search(v, key)

    recursive_search(schema, key)
    return schema
