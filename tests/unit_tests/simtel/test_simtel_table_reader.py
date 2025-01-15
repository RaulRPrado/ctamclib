#!/usr/bin/python3

import logging
from unittest import mock

import astropy.units as u
import pytest

import simtools.simtel.simtel_table_reader as simtel_table_reader


@pytest.fixture
def spe_test_file():
    return "tests/resources/spe_LST_2022-04-27_AP2.0e-4.dat"


@pytest.fixture
def spe_meta_test_comment():
    return "Norm_spe processing of single-p.e. response."


def test_read_simtel_data(spe_test_file, spe_meta_test_comment):
    """Test reading of sim_telarray table file into strings."""

    data, meta, n_columns, n_dim = simtel_table_reader._read_simtel_data(spe_test_file)

    assert isinstance(meta, str)
    assert spe_meta_test_comment in meta

    assert isinstance(data, list)
    assert len(data) > 0

    assert n_dim is None
    assert n_columns == 3


@mock.patch("simtools.utils.general.read_file_encoded_in_utf_or_latin")
def test_read_simtel_data_rpol(mock_read_file_encoded_in_utf_or_latin):
    mock_read_file_encoded_in_utf_or_latin.return_value = [
        "#@RPOL@[ANGLE=] 2",
        "ANGLE=     0              10              20",
        "# Blabla",
        "200     0.9353535       0.9349322       0.9332631",
    ]
    rows, meta, n_columns, n_dim_axis = simtel_table_reader._read_simtel_data("test_file")
    assert n_columns == 4
    assert n_dim_axis == ["0", "10", "20"]
    assert "Blabla" in meta
    assert len(rows) == 1


def test_read_simtel_table_to_table(spe_test_file, spe_meta_test_comment):
    """Test reading of sim_telarray pm_photoelectron_spectrum table file into astropy table."""

    parameter_name = "pm_photoelectron_spectrum"

    table = simtel_table_reader.read_simtel_table(parameter_name, spe_test_file)

    assert len(table) == 2101
    assert "amplitude" in table.columns
    assert "response" in table.columns
    assert "response_with_ap" in table.columns
    assert table["amplitude"].unit is None
    assert table["response"].unit is None
    assert table["response_with_ap"].unit is None
    assert table.meta["Name"] == parameter_name
    assert table.meta["File"] == spe_test_file
    assert spe_meta_test_comment in table.meta["Context_from_sim_telarray"]

    with pytest.raises(
        ValueError, match="Unsupported parameter for sim_telarray table reading: not_a_parameter"
    ):
        simtel_table_reader.read_simtel_table("not_a_parameter", spe_test_file)

    with mock.patch(
        "simtools.simtel.simtel_table_reader._read_simtel_data_for_atmospheric_transmission"
    ) as mock_read:
        simtel_table_reader.read_simtel_table("atmospheric_transmission", "test_file")
        mock_read.assert_called_once()


def test_data_simple_columns():
    columns = [
        "pm_photoelectron_spectrum",
        "quantum_efficiency",
        "camera_filter",
        "lightguide_efficiency_vs_wavelength",
        "lightguide_efficiency_vs_incidence_angle",
        "nsb_reference_spectrum",
        "atmospheric_profile",
    ]
    for column in columns:
        column_list, description = simtel_table_reader._data_columns(column, 2, None)
        assert isinstance(column_list, list)
        assert isinstance(description, str)


def test_data_columns_mirror_reflectivity():

    columns, description = simtel_table_reader._data_columns_mirror_reflectivity(2, ["0", "20"])
    assert columns == [
        {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
        {"name": "reflectivity_0deg", "description": "Mirror reflectivity at 0 deg", "unit": None},
        {
            "name": "reflectivity_20deg",
            "description": "Mirror reflectivity at 20 deg",
            "unit": None,
        },
    ]
    assert description == "Mirror reflectivity"
    for n_columns in [2, 3, 4]:
        columns, _ = simtel_table_reader._data_columns_mirror_reflectivity(n_columns, None)
        assert len(columns) == n_columns


def test_data_columns_pulse_shape():
    columns, description = simtel_table_reader._data_columns_pulse_shape(2)
    assert columns == [
        {"name": "time", "description": "Time", "unit": "ns"},
        {"name": "amplitude", "description": "Amplitude", "unit": None},
    ]
    assert description == "Pulse shape"
    for n_columns in [2, 3]:
        columns, _ = simtel_table_reader._data_columns_pulse_shape(n_columns)
        assert len(columns) == n_columns


@mock.patch("simtools.simtel.simtel_table_reader._data_columns_pulse_shape")
@mock.patch("simtools.simtel.simtel_table_reader._data_columns_mirror_reflectivity")
def test_data_columns(mock_data_columns_mirror_reflectivity, mock_data_columns_pulse_shape):
    simtel_table_reader._data_columns("mirror_reflectivity", 2, ["0", "20"])
    mock_data_columns_mirror_reflectivity.assert_called_once_with(2, ["0", "20"])

    simtel_table_reader._data_columns("fadc_pulse_shape", 4, None)
    mock_data_columns_pulse_shape.assert_called_once_with(4)


def test_read_simtel_data_for_atmospheric_transmission(caplog):
    test_data = """
#============================
#
# MODTRAN options as follows:
#
# Atmospheric model: 1 (Tropical atmosphere)
# Haze: 3 (NAVY MARITIME extinction, VIS is wind and humidity dependent)
# Season: 2
# Vulcanic dust: 0
# Current wind speed: 0.100000 m/s
# 24 h average wind speed: 0.100000 m/s
# Zenith angle:  0.00 deg
# End altitude: 2.156 km
# Ground altitude: 0.000 km
#
#============================

# H2= 2.156, H1=    2.206     2.256     2.356     2.456     2.656     2.856     3.156     3.656     4.156     4.500     5.000     5.500     6.000     7.000     8.000     9.000    10.000    11.000    12.000    13.000    14.000    15.000    16.000    18.000    20.000    22.000    24.000    26.000    28.000    30.000    32.500    35.000    37.500    40.000    45.000    50.000    60.000    70.000    80.000    100.000
    200       0.264958  0.528056  1.048710  1.562035  2.566957  3.543001  4.960817  7.223014  9.210340  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00
    201       0.266000  0.530126  1.052809  1.568132  2.576958  3.556853  4.980130  7.264430  9.210340  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00
    202       0.266374  0.530868  1.054293  1.570320  2.580533  3.561308  4.990833  7.264430  9.210340  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00
    203       0.265970  0.530056  1.052662  1.567903  2.576554  3.555939  4.976593  7.264430  9.210340  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00
    204       0.264762  0.527657  1.047906  1.560804  2.564867  3.539938  4.956099  7.197119  9.210340  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00  99999.00
    """

    mock_string = "simtools.utils.general.read_file_encoded_in_utf_or_latin"

    with mock.patch(mock_string, return_value=test_data.splitlines()):
        table = simtel_table_reader._read_simtel_data_for_atmospheric_transmission("dummy_path")

    assert len(table) == 45
    assert "wavelength" in table.colnames
    assert "altitude" in table.colnames
    assert "extinction" in table.colnames
    assert table.meta["Name"] == "atmospheric_transmission"
    assert table.meta["File"] == "dummy_path"
    assert "MODTRAN options as follows:" in table.meta["Context_from_sim_telarray"]
    assert table["wavelength"][0] == 200
    assert table["altitude"][0] == 2.206
    assert table["extinction"][0] == 0.264958
    assert isinstance(table.meta["observatory_level"], u.Quantity)
    assert table.meta["observatory_level"] == 2.156 * u.km

    test_data += "\n   # not a comment"  # invalid, as comment not at beginning of line
    with mock.patch(mock_string, return_value=test_data.splitlines()):
        with caplog.at_level(logging.DEBUG):
            simtel_table_reader._read_simtel_data_for_atmospheric_transmission("dummy_path")
        assert "Skipping malformed line" in caplog.text

    test_data = "\n".join(line for line in test_data.splitlines() if "H1=" not in line)
    with mock.patch(mock_string, return_value=test_data.splitlines()):
        with pytest.raises(ValueError, match=r"^Header with 'H1='"):
            simtel_table_reader._read_simtel_data_for_atmospheric_transmission("dummy_path")
