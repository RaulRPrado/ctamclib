#!/usr/bin/python3
r"""
    Validate a file or files in a directory using a schema.

    Input files can be metadata, schema, or data files in yaml, json, or ecsv format.

    Command line arguments
    ----------------------
    file_name (str)
      input file to be validated
    model_parameters_directory (str)
        directory with json files of model parameters to be validated
    schema (str)
      schema file (jsonschema format) used for validation
    data_type (str)
        type of input data (allowed types: metadata, schema, data)

    Raises
    ------
    FileNotFoundError
      if file to be validated is not found

    Example
    -------

    .. code-block:: console

        simtools-validate-file-using-schema \\
         --file_name tests/resources/MLTdata-preproduction.meta.yml \\
         --schema simtools/schemas/metadata.metaschema.yml \\
         --data_type metadata

"""

import logging
import re
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH
from simtools.data_model import metadata_collector, schema, validate_data


def _parse(label, description):
    """
    Parse command line configuration.

    Parameters
    ----------
    label (str)
        application label
    description (str)
        application description

    Returns
    -------
    config (Configurator)
        application configuration

    """
    config = configurator.Configurator(label=label, description=description)
    group = config.parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file_name", help="File to be validated")
    group.add_argument(
        "--file_directory",
        help=(
            "Directory with json files to be validated. "
            "If no schema file is provided, the assumption is that model "
            "parameters are validated and the schema files are taken from "
            f"{MODEL_PARAMETER_SCHEMA_PATH}."
        ),
    )
    config.parser.add_argument("--schema", help="Json schema file", required=False)
    config.parser.add_argument(
        "--data_type",
        help="Type of input data",
        choices=["metadata", "schema", "data", "model_parameter"],
        default="data",
    )
    config.parser.add_argument(
        "--require_exact_data_type",
        help="Require exact data type for validation",
        action="store_true",
    )
    return config.initialize(paths=False)


def _get_schema_file_name(args_dict, data_dict=None):
    """
    Get schema file name from metadata, data dict, or from command line argument.

    Parameters
    ----------
    args_dict (dict)
        command line arguments
    data_dict (dict)
        dictionary with metaschema information

    Returns
    -------
    schema_file: str
        schema file name

    """
    schema_file = args_dict.get("schema")
    if schema_file is None and data_dict is not None:
        schema_file = data_dict.get("meta_schema_url")
    if schema_file is None:
        metadata = metadata_collector.MetadataCollector(
            None, metadata_file_name=args_dict["file_name"]
        )
        schema_file = metadata.get_data_model_schema_file_name()
    return schema_file


def _get_json_file_list(file_directory=None, file_name=None):
    """Return list of json files in a directory."""
    file_list = []
    if file_directory is not None:
        file_list = list(Path(file_directory).rglob("*.json"))
        if not file_list:
            raise FileNotFoundError(f"No files found in {file_directory}")
    elif file_name is not None:
        file_list = [file_name]

    return file_list


def validate_dict_using_schema(args_dict, logger):
    """
    Validate a schema file (or several files) given in yaml or json format.

    Schema is either given as command line argument, read from the meta_schema_url or from
    the metadata section of the data dictionary.

    """
    for file_name in _get_json_file_list(
        args_dict.get("file_directory"), args_dict.get("file_name")
    ):
        try:
            data = gen.collect_data_from_file(file_name=file_name)
        except FileNotFoundError as exc:
            logger.error(f"Error reading schema file from {file_name}")
            raise exc
        schema.validate_dict_using_schema(data, _get_schema_file_name(args_dict, data))
        logger.info(f"Successful validation of file {file_name}")


def validate_data_files(args_dict, logger):
    """Validate data files."""
    if args_dict.get("file_directory") is not None:
        tmp_args_dict = {}
        for file_name in _get_json_file_list(args_dict.get("file_directory")):
            tmp_args_dict["file_name"] = file_name
            parameter_name = re.sub(r"-\d+\.\d+\.\d+", "", file_name.stem)
            schema_file = schema.get_model_parameter_schema_file(f"{parameter_name}")
            tmp_args_dict["schema"] = schema_file
            tmp_args_dict["data_type"] = "model_parameter"
            tmp_args_dict["require_exact_data_type"] = args_dict["require_exact_data_type"]
            validate_data_file(tmp_args_dict, logger)
    else:
        validate_data_file(args_dict, logger)


def validate_data_file(args_dict, logger):
    """Validate a data file (e.g., in ecsv, json, yaml format)."""
    data_validator = validate_data.DataValidator(
        schema_file=_get_schema_file_name(args_dict),
        data_file=args_dict["file_name"],
        check_exact_data_type=args_dict["require_exact_data_type"],
    )
    data_validator.validate_and_transform(
        is_model_parameter=(args_dict["data_type"].lower() == "model_parameter")
    )

    logger.info(f"Successful validation of data file {args_dict['file_name']}")


def validate_metadata(args_dict, logger):
    """Validate metadata."""
    # metadata_collector runs the metadata validation by default, no need to do anything else
    metadata_collector.MetadataCollector(None, metadata_file_name=args_dict["file_name"])
    logger.info(f"Successful validation of metadata {args_dict['file_name']}")


def main():  # noqa: D103
    label = Path(__file__).stem
    args_dict, _ = _parse(
        label, description="Validate a file (metadata, schema, or data file) using a schema."
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    if args_dict["data_type"].lower() == "metadata":
        validate_metadata(args_dict, logger)
    elif args_dict["data_type"].lower() == "schema":
        validate_dict_using_schema(args_dict, logger)
    else:
        validate_data_files(args_dict, logger)


if __name__ == "__main__":
    main()
