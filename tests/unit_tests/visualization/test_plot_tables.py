#!/usr/bin/python3

from unittest import mock

import pytest

from simtools.visualization import plot_tables


@mock.patch("simtools.visualization.plot_tables.visualize")
@mock.patch("simtools.visualization.plot_tables.read_table_data")
def test_plot(mock_read_table_data, mock_visualize):
    config = {
        "tables": [
            {
                "label": "test_table",
                "file_name": "test_file",
                "type": "csv",
                "column_x": "x",
                "column_y": "y",
            }
        ]
    }
    output_file = "output.png"
    mock_data = {"test_table": "mock_data"}
    mock_read_table_data.return_value = mock_data
    mock_fig = mock.MagicMock()
    mock_visualize.plot_1d.return_value = mock_fig

    plot_tables.plot(config, output_file)

    mock_read_table_data.assert_called_once_with(config, None)
    mock_visualize.plot_1d.assert_called_once_with(mock_data, **config)
    mock_visualize.save_figure.assert_called_once_with(mock_fig, output_file)


@mock.patch("simtools.visualization.plot_tables.gen.get_structure_array_from_table")
@mock.patch("simtools.visualization.plot_tables.legacy_data_handler.read_legacy_data_as_table")
@mock.patch("simtools.visualization.plot_tables._read_table_from_model_database")
def test_read_table_data_from_file(
    mock_read_table_from_model_database,
    mock_read_legacy_data_as_table,
    mock_get_structure_array_from_table,
):
    config = {
        "tables": [
            {
                "label": "test_table",
                "file_name": "test_file",
                "type": "csv",
                "column_x": "x",
                "column_y": "y",
            }
        ]
    }
    db_config = None
    mock_table = mock.MagicMock()
    mock_read_legacy_data_as_table.return_value = mock_table
    mock_structure_array = mock.MagicMock()
    mock_get_structure_array_from_table.return_value = mock_structure_array

    result = plot_tables.read_table_data(config, db_config)

    mock_read_legacy_data_as_table.assert_called_once_with("test_file", "csv")
    mock_get_structure_array_from_table.assert_called_once_with(mock_table, ["x", "y"])
    assert result == {"test_table": mock_structure_array}


@mock.patch("simtools.visualization.plot_tables.gen.get_structure_array_from_table")
@mock.patch("simtools.visualization.plot_tables.legacy_data_handler.read_legacy_data_as_table")
@mock.patch("simtools.visualization.plot_tables._read_table_from_model_database")
def test_read_table_data_from_model_database(
    mock_read_table_from_model_database,
    mock_read_legacy_data_as_table,
    mock_get_structure_array_from_table,
):
    config = {
        "tables": [
            {
                "label": "test_table",
                "parameter": "test_parameter",
                "site": "test_site",
                "telescope": "test_telescope",
                "model_version": "test_version",
                "column_x": "x",
                "column_y": "y",
            }
        ]
    }
    db_config = None
    mock_table = mock.MagicMock()
    mock_read_table_from_model_database.return_value = mock_table
    mock_structure_array = mock.MagicMock()
    mock_get_structure_array_from_table.return_value = mock_structure_array

    result = plot_tables.read_table_data(config, db_config)

    mock_read_table_from_model_database.assert_called_once_with(config["tables"][0], db_config)
    mock_get_structure_array_from_table.assert_called_once_with(mock_table, ["x", "y"])
    assert result == {"test_table": mock_structure_array}


def test_read_table_data_no_table_data_defined():
    config = {"tables": [{"label": "test_table", "column_x": "x", "column_y": "y"}]}
    db_config = None

    with pytest.raises(ValueError, match="No table data defined in configuration."):
        plot_tables.read_table_data(config, db_config)


@mock.patch("simtools.visualization.plot_tables.TelescopeModel")
def test_read_table_from_model_database(mock_telescope_model_class):
    table_config = {
        "site": "test_site",
        "telescope": "test_telescope",
        "model_version": "test_version",
        "parameter": "test_parameter",
    }
    db_config = None
    mock_telescope_model = mock_telescope_model_class.return_value
    mock_table = mock.MagicMock()
    mock_telescope_model.get_model_file_as_table.return_value = mock_table

    result = plot_tables._read_table_from_model_database(table_config, db_config)

    mock_telescope_model_class.assert_called_once_with(
        site="test_site",
        telescope_name="test_telescope",
        model_version="test_version",
        mongo_db_config=db_config,
    )
    mock_telescope_model.get_model_file_as_table.assert_called_once_with("test_parameter")
    assert result == mock_table


@mock.patch("simtools.visualization.plot_tables.SiteModel")
def test_read_table_from_model_database_site(mock_site_model_class):
    table_config = {
        "site": "test_site",
        "model_version": "test_version",
        "parameter": "test_parameter",
    }
    db_config = None
    mock_site_model = mock_site_model_class.return_value
    mock_table = mock.MagicMock()
    mock_site_model.get_model_file_as_table.return_value = mock_table

    plot_tables._read_table_from_model_database(table_config, db_config)

    mock_site_model_class.assert_called_once_with(
        site="test_site",
        model_version="test_version",
        mongo_db_config=db_config,
    )


def test_read_table_and_normalize():
    config = {
        "tables": [
            {
                "file_name": "tests/resources//SinglePhe_spectrum_totalfit_19pixel-average_20200601.csv",
                "type": "legacy_lst_single_pe",
                "label": "test_table",
                "column_x": "amplitude",
                "column_y": "response",
                "normalize_y": True,
            }
        ]
    }
    data = plot_tables.read_table_data(config, None)
    assert isinstance(data, dict)
    assert data["test_table"]["response"].max() == 1.0
