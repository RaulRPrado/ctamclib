#!/usr/bin/python3
"""Plot tabular data."""

import numpy as np

import simtools.utils.general as gen
from simtools.io_operations import legacy_data_handler
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.visualization import visualize


def plot(config, output_file, db_config=None):
    """
    Plot tabular data from data or from model parameter files.

    Parameters
    ----------
    config: dict
        Configuration dictionary for plotting.
    output_file: str
        Output file.
    """
    data = read_table_data(config, db_config)

    fig = visualize.plot_1d(
        data,
        **config,
    )
    visualize.save_figure(fig, output_file)


def read_table_data(config, db_config):
    """
    Read table data from file or parameter database.

    Parameters
    ----------
    config: dict
        Configuration dictionary for plotting.

    Returns
    -------
    Dict
        Dict with table data (astropy tables).
    """
    data = {}

    for _config in config["tables"]:
        if "parameter" in _config:
            table = _read_table_from_model_database(_config, db_config)
        elif "file_name" in _config:
            table = legacy_data_handler.read_legacy_data_as_table(
                _config["file_name"], _config["type"]
            )
        else:
            raise ValueError("No table data defined in configuration.")
        if _config.get("normalize_y"):
            table[_config["column_y"]] = (
                table[_config["column_y"]] / table[_config["column_y"]].max()
            )
        if _config.get("select_values"):
            table = _select_values_from_table(
                table,
                _config["select_values"]["column_name"],
                _config["select_values"]["value"],
            )
        data[_config["label"]] = gen.get_structure_array_from_table(
            table, [_config["column_x"], _config["column_y"]]
        )
    return data


def _read_table_from_model_database(table_config, db_config):
    """
    Read table data from model parameter database.

    Parameters
    ----------
    table_config: dict
        Configuration dictionary for table data.

    Returns
    -------
    Table
        Astropy table.
    """
    if "telescope" in table_config:
        model = TelescopeModel(
            site=table_config["site"],
            telescope_name=table_config["telescope"],
            model_version=table_config["model_version"],
            mongo_db_config=db_config,
        )
    else:
        model = SiteModel(
            site=table_config["site"],
            model_version=table_config["model_version"],
            mongo_db_config=db_config,
        )
    return model.get_model_file_as_table(table_config["parameter"])


def _select_values_from_table(table, column_name, value):
    """Return a table with only the rows where column_name == value."""
    return table[np.isclose(table[column_name], value)]
