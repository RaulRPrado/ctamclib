import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from simtools.production_configuration.generate_production_grid import GridGeneration

# Sample data for testing
axes = [
    {
        "name": "energy",
        "range": (1e9, 1e14),
        "binning": 2,
        "scaling": "log",
        "distribution": "uniform",
        "unit": "TeV",
    },
    {
        "name": "azimuth",
        "range": (70, 80),
        "binning": 2,
        "scaling": "linear",
        "distribution": "uniform",
        "unit": "deg",
    },
    {
        "name": "zenith_angle",
        "range": (20, 30),
        "binning": 2,
        "scaling": "linear",
        "distribution": "uniform",
        "unit": "deg",
    },
]


ctao_data_level = "B"
science_case = "high_precision"
coordinate_system = "ra_dec"

latitude = 28.7622  # degrees
longitude = -17.8920  # degrees
observing_location = EarthLocation(lon=longitude * u.deg, lat=latitude * u.deg, height=2000 * u.m)
observing_time = Time("2017-09-16 00:00:00")

grid_gen = GridGeneration(
    axes, ctao_data_level, science_case, coordinate_system, observing_location, observing_time
)


def test_generate_grid():
    grid_points = grid_gen.generate_grid()
    assert isinstance(grid_points, list)
    assert len(grid_points) > 0
    assert all(isinstance(point, dict) for point in grid_points)
    assert all("energy" in point for point in grid_points)
    assert all("azimuth" in point for point in grid_points)
    assert all("zenith_angle" in point for point in grid_points)


def test_generate_power_law_values():
    values = grid_gen.generate_power_law_values((1e9, 1e12), 10, 3)
    assert len(values) == 10
    assert values[0] >= 1e9
    assert values[-1] <= 1e12


def test_adjust_axis_range():
    adjusted_range = grid_gen.adjust_axis_range((1e9, 1e12), "energy")
    assert adjusted_range == (1e9, 1111111111111.111)


def test_convert_altaz_to_radec():
    alt, az = 45.0 * u.deg, 30.0 * u.deg
    radec = grid_gen.convert_altaz_to_radec(alt, az)
    assert isinstance(radec, SkyCoord)
    assert np.isclose(radec.ra.deg, 24.322823, atol=1e-5)
    assert np.isclose(radec.dec.deg, 61.203, atol=1e-5)


def test_convert_coordinates():
    grid_points = [
        {"zenith_angle": 30 * u.deg, "azimuth": 45 * u.deg},
        {"zenith_angle": 20 * u.deg, "azimuth": 60 * u.deg},
    ]
    converted_points = grid_gen.convert_coordinates(grid_points)

    assert "ra" in converted_points[0]
    assert "dec" in converted_points[0]
    assert isinstance(converted_points[0]["ra"], u.Quantity)
    assert isinstance(converted_points[0]["dec"], u.Quantity)
    assert converted_points[0]["ra"].unit == u.deg
    assert converted_points[0]["dec"].unit == u.deg
