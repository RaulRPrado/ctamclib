#!/usr/bin/python3

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import simtools.db.db_model_upload as db_model_upload


@pytest.fixture
def iter_dir():
    return "simtools.db.db_model_upload.Path.iterdir"


@pytest.fixture
def is_dir():
    return "simtools.db.db_model_upload.Path.is_dir"


@patch("simtools.db.db_model_upload.gen.collect_data_from_file")
def test_add_values_from_json_to_db(mock_collect_data_from_file):
    mock_collect_data_from_file.return_value = {
        "parameter": "test_param",
        "parameter_version": "1.0",
    }
    mock_db = Mock()
    file = "test_file.json"
    collection = "test_collection"
    db_name = "test_db"
    file_prefix = "test_prefix"

    db_model_upload.add_values_from_json_to_db(file, collection, mock_db, db_name, file_prefix)

    mock_collect_data_from_file.assert_called_once_with(file_name=file)
    mock_db.add_new_parameter.assert_called_once_with(
        db_name=db_name,
        par_dict={"parameter": "test_param", "parameter_version": "1.0"},
        collection_name=collection,
        file_prefix=file_prefix,
    )


@patch("simtools.db.db_model_upload.gen.collect_data_from_file")
def test_read_production_table(mock_collect_data_from_file):
    mock_collect_data_from_file.return_value = {
        "parameters": {"LSTN-design": "param_value_1", "LSTN-01": "param_value_2"},
        "design_model": {"LSTN-design": "design_value_1", "LSTN-01": "design_value_2"},
    }
    model_dict = {}
    file = Mock()
    file.stem = "LSTN-design"
    model_name = "test_model"

    db_model_upload._read_production_table(model_dict, file, model_name)

    assert "LSTN-design" in model_dict["telescopes"]["parameters"]
    assert model_dict["telescopes"]["parameters"]["LSTN-design"] == "param_value_1"
    assert model_dict["telescopes"]["design_model"]["LSTN-design"] == "design_value_1"

    file.stem = "MSTx-NectarCam"
    with pytest.raises(KeyError, match="MSTx-NectarCam"):
        db_model_upload._read_production_table(model_dict, file, model_name)

    file.stem = "configuration_corsika"
    mock_collect_data_from_file.return_value = {"parameters": "config_param_value"}

    db_model_upload._read_production_table(model_dict, file, model_name)

    assert model_dict["configuration_corsika"]["parameters"] == "config_param_value"


@patch("simtools.db.db_model_upload._read_production_table")
def test_add_production_tables_to_db(
    mock_read_production_table, tmp_test_directory, caplog, iter_dir, is_dir
):
    mock_db = Mock()
    args_dict = {"input_path": tmp_test_directory, "db_name": "test_db"}
    input_path = Path(args_dict["input_path"])
    model_dir = input_path / "model_version_1"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "file1.json").touch()
    (model_dir / "MSTS-02.json").touch()

    mock_read_production_table.side_effect = lambda model_dict, file, _: model_dict.update(
        {
            "telescopes": {
                "parameters": {"MSTS-02": "param_value"},
                "design_model": {"MSTS-02": "MSTx-FlashCam"},
            }
        }
    )

    with patch(iter_dir, return_value=[model_dir]):
        with patch(is_dir, return_value=True):
            db_model_upload.add_production_tables_to_db(args_dict, mock_db)

    assert mock_read_production_table.call_count == 2
    mock_db.add_production_table.assert_called_once_with(
        db_name="test_db",
        production_table={
            "parameters": {"MSTS-02": "param_value"},
            "design_model": {"MSTS-02": "MSTx-FlashCam"},
        },
    )

    mock_read_production_table.side_effect = lambda model_dict, file, _: model_dict.update(
        {"telescopes": {"parameters": {}, "design_model": {}}}
    )
    with caplog.at_level("INFO"):
        db_model_upload.add_production_tables_to_db(args_dict, mock_db)
    assert "No production table for telescopes in model version model_version_1" in caplog.text


@patch("simtools.db.db_model_upload.add_values_from_json_to_db")
def test_add_model_parameters_to_db(
    mock_add_values_from_json_to_db, tmp_test_directory, iter_dir, is_dir
):
    mock_db = Mock()
    args_dict = {"input_path": tmp_test_directory, "db_name": "test_db"}
    input_path = Path(args_dict["input_path"])
    array_element_dir = input_path / "LSTS-01"
    array_element_dir.mkdir(parents=True, exist_ok=True)
    (array_element_dir / "num_gains-0.1.0.json").touch()
    (array_element_dir / "mirror_list-0.2.1.json").touch()

    with patch(iter_dir, return_value=[array_element_dir]):
        with patch(is_dir, return_value=True):
            db_model_upload.add_model_parameters_to_db(args_dict, mock_db)

    mock_add_values_from_json_to_db.assert_any_call(
        file=array_element_dir / "num_gains-0.1.0.json",
        collection="telescopes",
        db=mock_db,
        db_name="test_db",
        file_prefix=input_path / "Files",
    )
    mock_add_values_from_json_to_db.assert_any_call(
        file=array_element_dir / "mirror_list-0.2.1.json",
        collection="telescopes",
        db=mock_db,
        db_name="test_db",
        file_prefix=input_path / "Files",
    )
    assert mock_add_values_from_json_to_db.call_count == 2


@patch("simtools.db.db_model_upload.add_values_from_json_to_db")
def test_add_model_parameters_to_db_skip_files_collection(
    mock_add_values_from_json_to_db, tmp_test_directory, iter_dir, is_dir
):
    mock_db = Mock()
    args_dict = {"input_path": tmp_test_directory, "db_name": "test_db"}
    input_path = Path(args_dict["input_path"])
    files_dir = input_path / "Files"
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / "file1.json").touch()

    with patch(iter_dir, return_value=[files_dir]):
        with patch(is_dir, return_value=True):
            db_model_upload.add_model_parameters_to_db(args_dict, mock_db)

    mock_add_values_from_json_to_db.assert_not_called()
