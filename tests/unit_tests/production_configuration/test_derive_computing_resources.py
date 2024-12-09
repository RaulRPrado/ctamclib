from unittest.mock import mock_open

import astropy.units as u
import pytest
import yaml

from simtools.production_configuration.derive_computing_resources import ResourceEstimator

BUILTINS_OPEN = "builtins.open"
VALIDATE_SITE_NAME = "simtools.utils.names.validate_site_name"
DUMMY_YAML_FILE = "dummy.yaml"

# Define the shared YAML file data as a constant
FILE_DATA = yaml.dump(
    {
        "example_site": {
            "Zenith": {
                30.0: {
                    "compute_per_event": {
                        "value": 1e-6,
                        "unit": "hr",
                    },
                    "storage_per_event": {
                        "value": 1e-7,
                        "unit": "GB",
                    },
                },
                45.0: {
                    "compute_per_event": {
                        "value": 2e-6,
                        "unit": "hr",
                    },
                    "storage_per_event": {
                        "value": 2e-7,
                        "unit": "GB",
                    },
                },
            }
        }
    }
)


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    grid_point_config = {
        "azimuth": 60.0,
        "elevation": 45.0,
        "night_sky_background": 0.3,
    }

    simulation_params = {"number_of_events": 1e9, "site": "example_site"}

    existing_data = [
        {
            "azimuth": 60.0,
            "elevation": 45.0,
            "nsb": 0.3,
            "compute_total": {
                "value": 5000.0,
                "unit": "hr",
            },
            "storage_total": {
                "value": 500,
                "unit": "GB",
            },
            "events": 1e9,
        },
    ]

    lookup_table = {
        "example_site": {
            "Zenith": {
                30.0: {
                    "compute_per_event": {
                        "value": 1e-6,
                        "unit": "hr",
                    },
                    "storage_per_event": {
                        "value": 1e-7,
                        "unit": "GB",
                    },
                },
                45.0: {
                    "compute_per_event": {
                        "value": 2e-6,
                        "unit": "hr",
                    },
                    "storage_per_event": {
                        "value": 2e-7,
                        "unit": "GB",
                    },
                },
            }
        }
    }

    return grid_point_config, simulation_params, existing_data, lookup_table


@pytest.mark.parametrize("file_data", [FILE_DATA], ids=["example_site_lookup_table"])
def test_estimate_resources_interpolation(sample_data, file_data, monkeypatch):
    """Test resource estimation with interpolation."""
    grid_point_config, simulation_params, existing_data, _ = sample_data

    monkeypatch.setattr(BUILTINS_OPEN, mock_open(read_data=file_data))
    monkeypatch.setattr(VALIDATE_SITE_NAME, lambda x: "example_site")

    estimator = ResourceEstimator(
        grid_point=grid_point_config,
        simulation_params=simulation_params,
        existing_data=existing_data,
        lookup_file=DUMMY_YAML_FILE,
    )

    expected_resources = {
        "compute_total": u.Quantity(5000.0, u.hour),
        "storage_total": u.Quantity(500.0, u.GB),
    }

    resources = estimator.estimate_resources()
    assert resources == expected_resources


@pytest.mark.parametrize("file_data", [FILE_DATA], ids=["example_site_lookup_table"])
def test_guess_resources_per_event(sample_data, file_data, monkeypatch):
    """Test resource estimation based on guessed resources per event."""
    grid_point_config, simulation_params, _, lookup_table = sample_data

    monkeypatch.setattr(BUILTINS_OPEN, mock_open(read_data=file_data))
    monkeypatch.setattr(VALIDATE_SITE_NAME, lambda x: "example_site")

    estimator = ResourceEstimator(
        grid_point=grid_point_config,
        simulation_params=simulation_params,
        existing_data=None,
        lookup_file=DUMMY_YAML_FILE,
    )

    expected_resources = {
        "compute": u.Quantity(2000.0, u.hour),
        "storage": u.Quantity(200.0, u.GB),
    }

    resources = estimator.estimate_resources()

    assert resources == expected_resources


@pytest.mark.parametrize("file_data", [FILE_DATA], ids=["example_site_lookup_table"])
def test_load_lookup_table(sample_data, file_data, monkeypatch):
    """Test loading of the lookup table from a YAML file."""
    grid_point_config, simulation_params, _, lookup_table = sample_data

    monkeypatch.setattr(BUILTINS_OPEN, mock_open(read_data=file_data))
    monkeypatch.setattr(VALIDATE_SITE_NAME, lambda x: "example_site")

    estimator = ResourceEstimator(
        grid_point=grid_point_config,
        simulation_params=simulation_params,
        lookup_file=DUMMY_YAML_FILE,
    )

    expected_lookup_table = {
        "example_site": {
            "Zenith": {
                30.0: {
                    "compute_per_event": u.Quantity(1e-6, u.hr),
                    "storage_per_event": u.Quantity(1e-7, u.GB),
                },
                45.0: {
                    "compute_per_event": u.Quantity(2e-6, u.hr),
                    "storage_per_event": u.Quantity(2e-7, u.GB),
                },
            }
        }
    }
    assert estimator.lookup_table == expected_lookup_table


@pytest.mark.parametrize("file_data", [FILE_DATA], ids=["example_site_lookup_table"])
def test_interpolate_resources(sample_data, file_data, monkeypatch):
    """Test direct interpolation of resources from existing data."""
    grid_point_config, simulation_params, existing_data, _ = sample_data

    monkeypatch.setattr(BUILTINS_OPEN, mock_open(read_data=file_data))
    monkeypatch.setattr(VALIDATE_SITE_NAME, lambda x: "example_site")

    estimator = ResourceEstimator(
        grid_point=grid_point_config,
        simulation_params=simulation_params,
        existing_data=existing_data,
        lookup_file=DUMMY_YAML_FILE,
    )

    number_of_events = 1e9
    resources = estimator.interpolate_resources(number_of_events)

    expected_resources = {
        "compute_total": u.Quantity(5000.0, u.hour),
        "storage_total": u.Quantity(500.0, u.GB),
    }

    assert resources == expected_resources
