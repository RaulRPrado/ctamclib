import logging
from pathlib import Path

import jsonschema
import pytest
import yaml

from simtools.constants import MODEL_PARAMETER_DESCRIPTION_METASCHEMA, MODEL_PARAMETER_METASCHEMA
from simtools.data_model import metadata_model
from simtools.utils import general as gen


def test_get_default_metadata_dict():
    _top_meta = metadata_model.get_default_metadata_dict()

    assert isinstance(_top_meta, dict)
    assert len(_top_meta) > 0

    assert "VERSION" in _top_meta["CTA"]["REFERENCE"]
    assert _top_meta["CTA"]["REFERENCE"]["VERSION"] == "1.0.0"
    assert _top_meta["CTA"]["CONTACT"]["ORGANIZATION"] == "CTAO"


def test_load_schema(caplog, tmp_test_directory):
    _metadata_schema, _ = metadata_model._load_schema()
    assert isinstance(_metadata_schema, dict)
    assert len(_metadata_schema) > 0

    with pytest.raises(FileNotFoundError):
        metadata_model._load_schema(schema_file="not_existing_file")

    # schema versions
    with pytest.raises(ValueError, match=r"^Schema version not given in"):
        metadata_model._load_schema(MODEL_PARAMETER_METASCHEMA)

    _schema_1, _ = metadata_model._load_schema(MODEL_PARAMETER_METASCHEMA, "0.1.0")
    assert _schema_1["version"] == "0.1.0"
    _schema_2, _ = metadata_model._load_schema(MODEL_PARAMETER_METASCHEMA, "0.2.0")
    assert _schema_2["version"] == "0.2.0"

    with pytest.raises(ValueError, match=r"^Schema version 0.2 not found in"):
        metadata_model._load_schema(MODEL_PARAMETER_METASCHEMA, "0.2")

    # test a single doc yaml file (write a temporary schema file; to make sure it is a single doc)
    tmp_schema_file = Path(tmp_test_directory) / "schema.yml"
    with open(tmp_schema_file, "w", encoding="utf-8") as f:
        yaml.dump(_schema_2, f)

    with caplog.at_level(logging.WARNING):
        _schema_3, _ = metadata_model._load_schema(tmp_schema_file, "0.3.0")
    assert "Schema version 0.3.0 does not match 0.2.0" in caplog.text


def test_validate_schema(tmp_test_directory):
    sample_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        "required": ["name", "age"],
    }

    schema_file = Path(tmp_test_directory) / "schema.yml"
    with open(schema_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_schema, f)

    # sample data dictionary to be validated
    data = {"name": "John", "age": 30}

    metadata_model.validate_schema(data, schema_file)

    invalid_data = {"name": "Alice", "age": "Thirty"}
    with pytest.raises(jsonschema.exceptions.ValidationError):
        metadata_model.validate_schema(invalid_data, schema_file)


def test_validate_schema_astropy_units(caplog):
    success_string = "Successful validation of data using schema from"

    _dict_1 = gen.collect_data_from_file(file_name="tests/resources/num_gains.schema.yml")
    with caplog.at_level(logging.DEBUG):
        metadata_model.validate_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # m and cm
    _dict_1["data"][0]["unit"] = "m"
    with caplog.at_level(logging.DEBUG):
        metadata_model.validate_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = "cm"
    with caplog.at_level(logging.DEBUG):
        metadata_model.validate_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # combined units
    _dict_1["data"][0]["unit"] = "cm/s"
    with caplog.at_level(logging.DEBUG):
        metadata_model.validate_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = "km/ s"
    with caplog.at_level(logging.DEBUG):
        metadata_model.validate_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # dimensionless
    _dict_1["data"][0]["unit"] = "dimensionless"
    with caplog.at_level(logging.DEBUG):
        metadata_model.validate_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text
    _dict_1["data"][0]["unit"] = ""
    with caplog.at_level(logging.DEBUG):
        metadata_model.validate_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )
    assert success_string in caplog.text

    # not good
    _dict_1["data"][0]["unit"] = "not_a_unit"
    with pytest.raises(ValueError, match="'not_a_unit' is not a valid Unit"):
        metadata_model.validate_schema(
            data=_dict_1, schema_file=MODEL_PARAMETER_DESCRIPTION_METASCHEMA
        )


def test_resolve_references():
    yaml_data = {
        "example_data": {
            "example_object": {"type": "object", "properties": {"INSTRUMENT": {"type": "string"}}},
            "another_object": {
                "type": "object",
                "properties": {
                    "INSTRUMENT": {"$ref": "#/example_data/example_object/properties/INSTRUMENT"}
                },
            },
        }
    }

    expected_result = {
        "example_data": {
            "example_object": {"type": "object", "properties": {"INSTRUMENT": {"type": "string"}}},
            "another_object": {"type": "object", "properties": {"INSTRUMENT": {"type": "string"}}},
        }
    }

    assert metadata_model._resolve_references(yaml_data) == expected_result


def test_add_array_elements():

    test_dict_1 = {"data": {"InstrumentTypeElement": {"enum": ["LSTN", "MSTN"]}}}
    test_dict_added = metadata_model._add_array_elements("InstrumentTypeElement", test_dict_1)
    assert len(test_dict_added["data"]["InstrumentTypeElement"]["enum"]) > 2
    test_dict_2 = {"data": {"InstrumentTypeElement": {"not_the_right_enum": ["LSTN", "MSTN"]}}}
    test_dict_added_2 = metadata_model._add_array_elements("InstrumentTypeElement", test_dict_2)
    assert len(test_dict_added_2["data"]["InstrumentTypeElement"]["enum"]) > 2


def test_fill_defaults(caplog):
    schema = {
        "CTA": {
            "properties": {
                "CONTACT": {
                    "type": "object",
                    "properties": {
                        "organization": {"type": "string", "default": "CTA"},
                        "number": {"type": "integer", "default": 30},
                    },
                },
                "DOCUMENTS": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "default": "a_document"},
                            "id": {"type": "integer", "default": 55},
                        },
                    },
                },
                "NO_DEFAULT": {
                    "type": "object",
                    "properties": {
                        "string_without_default": {
                            "type": "string",
                        },
                    },
                },
                "NO_DEFAULT_LIST": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "string_without_default": {
                                "type": "string",
                            },
                        },
                    },
                },
            }
        }
    }
    expected_result = {
        "CTA": {
            "CONTACT": {"organization": "CTA", "number": 30},
            "DOCUMENTS": [{"name": "a_document", "id": 55}],
            "NO_DEFAULT": {},
            "NO_DEFAULT_LIST": [{}],
        }
    }

    assert metadata_model._fill_defaults(schema) == expected_result

    schema = {
        "CTA": {
            "CONTACT": {
                "type": "object",
                "no_properties": {
                    "organization": {"type": "string", "default": "CTA"},
                    "number": {"type": "integer", "default": 30},
                },
            },
        }
    }

    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyError):
            metadata_model._fill_defaults(schema)

    assert "Missing 'properties' key in schema." in caplog.text
