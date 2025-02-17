#!/usr/bin/python3

import logging
from pathlib import Path

import jsonschema
import pytest
import yaml

from simtools.constants import (
    MODEL_PARAMETER_DESCRIPTION_METASCHEMA,
    MODEL_PARAMETER_METASCHEMA,
    MODEL_PARAMETER_SCHEMA_PATH,
)
from simtools.data_model import schema
from simtools.utils import general as gen


def test_get_get_model_parameter_schema_files(tmp_test_directory):

    par, files = schema.get_get_model_parameter_schema_files()
    assert len(files)
    assert files[0].is_file()
    assert "num_gains" in par

    # no files in the directory
    with pytest.raises(FileNotFoundError, match=r"^No schema files"):
        schema.get_get_model_parameter_schema_files(tmp_test_directory)

    # directory does not exist
    with pytest.raises(FileNotFoundError, match=r"^No schema files"):
        schema.get_get_model_parameter_schema_files("not_a_directory")


def test_get_model_parameter_schema_file():

    schema_file = str(schema.get_model_parameter_schema_file("num_gains"))

    assert str(MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml") in schema_file

    with pytest.raises(FileNotFoundError, match=r"^Schema file not found:"):
        schema.get_model_parameter_schema_file("not_a_parameter")


def test_get_model_parameter_schema_version():

    most_recent = schema.get_model_parameter_schema_version()
    assert most_recent == "0.3.0"

    assert schema.get_model_parameter_schema_version("0.2.0") == "0.2.0"
    assert schema.get_model_parameter_schema_version("0.1.0") == "0.1.0"

    with pytest.raises(ValueError, match=r"^Schema version 0.0.1 not found in"):
        schema.get_model_parameter_schema_version("0.0.1")


def test_validate_dict_using_schema(tmp_test_directory, caplog):

    with caplog.at_level(logging.WARNING):
        schema.validate_dict_using_schema(None, None)
    assert "No schema provided for validation of" in caplog.text

    sample_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        "meta_schema_url": "string",
        "required": ["name", "age"],
    }

    schema_file = Path(tmp_test_directory) / "schema.yml"
    with open(schema_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_schema, f)

    # sample data dictionary to be validated
    data = {"name": "John", "age": 30}

    schema.validate_dict_using_schema(data, schema_file)

    invalid_data = {"name": "Alice", "age": "Thirty"}
    with pytest.raises(jsonschema.exceptions.ValidationError):
        schema.validate_dict_using_schema(invalid_data, schema_file)

    # with valid meta_schema_url
    data["meta_schema_url"] = "https://github.com/gammasim/simtools"
    schema.validate_dict_using_schema(data, schema_file)

    data["meta_schema_url"] = "https://invalid_url"
    with pytest.raises(FileNotFoundError, match=r"^Meta schema URL does not exist:"):
        schema.validate_dict_using_schema(data, schema_file)


def test_validate_schema_astropy_units(caplog):
    success_string = "Successful validation of data using schema"

    _dict_1 = gen.collect_data_from_file(file_name="tests/resources/num_gains.schema.yml")
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # m and cm
    _dict_1["data"][0]["unit"] = "m"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = "cm"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # combined units
    _dict_1["data"][0]["unit"] = "cm/s"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = "km/ s"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # dimensionless
    _dict_1["data"][0]["unit"] = "dimensionless"
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = ""
    with caplog.at_level(logging.DEBUG):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # not good
    _dict_1["data"][0]["unit"] = "not_a_unit"
    with pytest.raises(ValueError, match="'not_a_unit' is not a valid Unit"):
        schema.validate_dict_using_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )


def test_load_schema(caplog, tmp_test_directory):
    _metadata_schema = schema.load_schema()
    assert isinstance(_metadata_schema, dict)
    assert len(_metadata_schema) > 0

    with pytest.raises(FileNotFoundError):
        schema.load_schema(schema_file="not_existing_file")

    # schema versions
    with pytest.raises(ValueError, match=r"^Schema version not given in"):
        schema.load_schema(MODEL_PARAMETER_METASCHEMA)

    _schema_1 = schema.load_schema(MODEL_PARAMETER_METASCHEMA, "0.1.0")
    assert _schema_1["version"] == "0.1.0"
    _schema_2 = schema.load_schema(MODEL_PARAMETER_METASCHEMA, "0.2.0")
    assert _schema_2["version"] == "0.2.0"

    with pytest.raises(ValueError, match=r"^Schema version 0.2 not found in"):
        schema.load_schema(MODEL_PARAMETER_METASCHEMA, "0.2")

    # test a single doc yaml file (write a temporary schema file; to make sure it is a single doc)
    tmp_schema_file = Path(tmp_test_directory) / "schema.yml"
    with open(tmp_schema_file, "w", encoding="utf-8") as f:
        yaml.dump(_schema_2, f)

    with caplog.at_level(logging.WARNING):
        schema.load_schema(tmp_schema_file, "0.3.0")
    assert "Schema version 0.3.0 does not match 0.2.0" in caplog.text


def test_add_array_elements():

    test_dict_1 = {"data": {"InstrumentTypeElement": {"enum": ["LSTN", "MSTN"]}}}
    test_dict_added = schema._add_array_elements("InstrumentTypeElement", test_dict_1)
    assert len(test_dict_added["data"]["InstrumentTypeElement"]["enum"]) > 2
    test_dict_2 = {"data": {"InstrumentTypeElement": {"not_the_right_enum": ["LSTN", "MSTN"]}}}
    test_dict_added_2 = schema._add_array_elements("InstrumentTypeElement", test_dict_2)
    assert len(test_dict_added_2["data"]["InstrumentTypeElement"]["enum"]) > 2
