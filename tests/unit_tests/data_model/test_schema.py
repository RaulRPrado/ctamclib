#!/usr/bin/python3

import pytest

from simtools.data_model import schema


def test_model_parameter_schema_files(tmp_test_directory):

    par, files = schema.model_parameter_schema_files()
    assert len(files)
    assert files[0].is_file()
    assert "num_gains" in par

    # no files in the directory
    with pytest.raises(FileNotFoundError, match=r"^No schema files"):
        schema.model_parameter_schema_files(tmp_test_directory)

    # directory does not exist
    with pytest.raises(FileNotFoundError, match=r"^No schema files"):
        schema.model_parameter_schema_files("not_a_directory")


def test_model_parameter_schema_file():

    schema_file = str(schema.model_parameter_schema_file("num_gains"))

    assert "simtools/schemas/model_parameters/num_gains.schema.yml" in schema_file

    with pytest.raises(FileNotFoundError, match=r"^Schema file not found:"):
        schema.model_parameter_schema_file("not_a_parameter")


def test_model_parameter_schema_version():

    most_recent = schema.model_parameter_schema_version()
    assert most_recent == "0.2.0"

    assert schema.model_parameter_schema_version("0.2.0") == "0.2.0"
    assert schema.model_parameter_schema_version("0.1.0") == "0.1.0"

    with pytest.raises(ValueError, match=r"^Schema version 0.0.1 not found in"):
        schema.model_parameter_schema_version("0.0.1")
