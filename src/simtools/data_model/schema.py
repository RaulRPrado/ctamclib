"""Module providing functionality to read schema."""

from pathlib import Path

import simtools.utils.general as gen
from simtools.constants import MODEL_PARAMETER_METASCHEMA, MODEL_PARAMETER_SCHEMA_PATH


def model_parameter_schema_files(schema_directory=MODEL_PARAMETER_SCHEMA_PATH):
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


def model_parameter_schema_file(parameter):
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


def model_parameter_schema_version(schema_version=None):
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
