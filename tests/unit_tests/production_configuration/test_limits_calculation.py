import astropy.units as u
import pytest
from astropy.table import Table

from simtools.production_configuration.limits_calculation import LimitCalculator


@pytest.fixture
def hdf5_file():
    angle_table = Table(
        data={"angle": [0.1, 0.2, 0.3] * u.deg},
        meta={
            "Title": "angle_to_observing_position__triggered_showers_",
            "x_bin_edges": [0, 0.1, 0.2, 0.3, 0.4],
            "x_bin_edges_units": "deg",
        },
    )
    event_table = Table(
        data={"weight": [1, 2, 3]},
        meta={
            "Title": "event_weight__ra3d__log10_e__",
            "x_bin_edges": [0, 1, 2, 3, 4],
            "y_bin_edges": [0, 1, 2, 3, 4],
            "x_bin_edges_units": "TeV",
            "y_bin_edges_units": "m",
        },
    )
    return [angle_table, event_table]


@pytest.fixture
def hdf5_file_no_units():
    angle_table = Table(
        data={"angle": [0.1, 0.2, 0.3]},
        meta={
            "Title": "angle_to_observing_position__triggered_showers_",
            "x_bin_edges": [0, 0.1, 0.2, 0.3, 0.4],
        },
    )
    event_table = Table(
        data={"weight": [1, 2, 3]},
        meta={
            "Title": "event_weight__ra3d__log10_e__",
            "x_bin_edges": [0, 1, 2, 3, 4],
            "y_bin_edges": [0, 1, 2, 3, 4],
        },
    )
    return [angle_table, event_table]


@pytest.fixture
def limit_calculator(hdf5_file):
    return LimitCalculator(hdf5_file)


@pytest.fixture
def limit_calculator_no_units(hdf5_file_no_units):
    return LimitCalculator(hdf5_file_no_units)


def test_compute_lower_energy_limit(limit_calculator):
    loss_fraction = 0.5
    lower_energy_limit = limit_calculator.compute_lower_energy_limit(loss_fraction)
    assert lower_energy_limit == 1 * u.TeV


def test_compute_lower_energy_limit_no_units(limit_calculator_no_units):
    loss_fraction = 0.5
    lower_energy_limit = limit_calculator_no_units.compute_lower_energy_limit(loss_fraction)
    assert lower_energy_limit == 1 * u.TeV  # LimitCalculator adds units


def test_compute_upper_radial_distance(limit_calculator):
    loss_fraction = 0.2
    upper_radial_distance = limit_calculator.compute_upper_radial_distance(loss_fraction)
    assert upper_radial_distance == 2 * u.m


def test_compute_upper_radial_distance_no_units(limit_calculator_no_units):
    loss_fraction = 0.2
    upper_radial_distance = limit_calculator_no_units.compute_upper_radial_distance(loss_fraction)
    assert upper_radial_distance == 2 * u.m  # LimitCalculator adds units


def test_compute_viewcone(limit_calculator):
    loss_fraction = 0.1
    viewcone = limit_calculator.compute_viewcone(loss_fraction)
    assert viewcone == 0 * u.deg


def test_compute_viewcone_no_units(limit_calculator_no_units):
    loss_fraction = 0.1
    viewcone = limit_calculator_no_units.compute_viewcone(loss_fraction)
    assert viewcone == 0 * u.deg  # LimitCalculator adds units
