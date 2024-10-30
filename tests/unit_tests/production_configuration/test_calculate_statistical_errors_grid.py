import numpy as np
import pytest
from astropy.io import fits

from simtools.production_configuration.calculate_statistical_errors_grid_point import (
    StatisticalErrorEvaluator,
)
from simtools.production_configuration.interpolation_handler import InterpolationHandler


@pytest.fixture
def test_fits_file(tmp_path):
    file_path = tmp_path / "test_file.fits"
    hdu_list = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(
                name="EVENTS",
                data=np.array(
                    [(1.0, 1.0, 180, 20.0), (2.0, 2.0, 180, 70.0)],
                    dtype=[
                        ("ENERGY", "f4"),
                        ("MC_ENERGY", "f4"),
                        ("PNT_AZ", "f4"),
                        ("PNT_ALT", "f4"),
                    ],
                ),
            ),
            fits.BinTableHDU(
                name="SIMULATED EVENTS",
                data=np.array(
                    [
                        (3e-03, 1.0e-02, 7e8),
                        (1e3, 2, 5e8),
                    ],
                    dtype=[("MC_ENERG_LO", "f4"), ("MC_ENERG_HI", "f4"), ("EVENTS", "f4")],
                ),
            ),
            fits.BinTableHDU(
                name="INFO",
                data=np.array(
                    [
                        ([0.0, 2800.0], [0.0, 10.0]),
                    ],
                    dtype=[
                        ("core_range", ">f8", (2,)),
                        ("viewcone", ">f8", (2,)),
                    ],
                ),
            ),
        ]
    )
    hdu_list.writeto(file_path, overwrite=True)
    return file_path


def test_initialization(test_fits_file):
    """Test the initialization of the StatisticalErrorEvaluator."""
    evaluator = StatisticalErrorEvaluator(file_path=test_fits_file, file_type="On-source")
    assert evaluator.file_path == test_fits_file
    assert evaluator.file_type == "On-source"
    assert isinstance(evaluator.data, dict)
    assert "event_energies" in evaluator.data


def test_calculate_error_eff_area(test_fits_file):
    """Test the calculation of effective area error."""
    evaluator = StatisticalErrorEvaluator(file_path=test_fits_file, file_type="On-source")
    evaluator.calculate_metrics()
    errors = evaluator.calculate_error_eff_area()
    assert "relative_errors" in errors
    assert len(errors["relative_errors"]) > 0


def test_calculate_error_energy_estimate_bdt_reg_tree(test_fits_file):
    """Test the calculation of energy estimate error."""
    evaluator = StatisticalErrorEvaluator(file_path=test_fits_file, file_type="On-source")
    evaluator.calculate_metrics()
    error, sigma, delta = evaluator.calculate_error_energy_estimate_bdt_reg_tree()
    # assert isinstance(error, float)
    assert isinstance(sigma, list)
    assert isinstance(delta, list)


def test_handle_missing_file():
    """Test error handling for missing file."""
    evaluator = StatisticalErrorEvaluator(file_path="missing_file.fits", file_type="On-source")
    assert evaluator.data == {}


def test_interpolation_handler(test_fits_file):
    """Test interpolation with the InterpolationHandler."""
    grid_point1 = (1, 180, 45, 0, 0.5)
    evaluator1 = StatisticalErrorEvaluator(
        file_path=test_fits_file, file_type="On-source", grid_point=grid_point1
    )
    grid_point2 = (1, 180, 60, 0, 0.5)

    evaluator2 = StatisticalErrorEvaluator(
        file_path=test_fits_file, file_type="On-source", grid_point=grid_point2
    )
    handler = InterpolationHandler([evaluator1, evaluator2])

    query_point = np.array([[1, 180, 50, 0, 0.5]])
    interpolated_values = handler.interpolate(query_point)
    assert interpolated_values.shape[0] == query_point.shape[0]
    # assert not np.isnan(interpolated_values).any() # requires actual file for interpolation

    query_point = np.array([[1e-3, 180, 40, 0, 0.5]])
    interpolated_threshold = handler.interpolate_energy_threshold(query_point)
    assert isinstance(interpolated_threshold, float)


def test_calculate_scaled_events(test_fits_file):
    """Test the calculation of scaled events for a specific grid point."""
    evaluator = StatisticalErrorEvaluator(file_path=test_fits_file, file_type="On-source")

    evaluator.grid_point = (1.5, 180, 45, 0, 0.5)

    evaluator.data = {
        "simulated_event_histogram": np.array([100, 200, 300]),
    }
    evaluator.error_eff_area = {
        "relative_errors": np.array([0.1, 0.2, 0.3]),
    }
    evaluator.metrics = {
        "error_eff_area": np.array([0.05, 0.1, 0.15]),
    }

    evaluator.create_bin_edges = lambda: np.array([1.0, 2.0, 3.0])

    scaled_events = evaluator.calculate_scaled_events()

    assert isinstance(scaled_events, float)
    assert scaled_events == pytest.approx(200.0, rel=1e-2)


def test_calculate_metrics(test_fits_file):
    """Test the calculation of metrics."""
    evaluator = StatisticalErrorEvaluator(file_path=test_fits_file, file_type="On-source")

    # Set metrics and mock errors
    evaluator.metrics = {
        "error_eff_area": 0.1,
        "error_sig_eff_gh": 0.2,
        "error_energy_estimate_bdt_reg_tree": 0.3,
        "error_gamma_ray_psf": 0.4,
        "error_image_template_methods": 0.5,
    }

    evaluator.calculate_error_eff_area = lambda: {"relative_errors": np.array([0.1, 0.2, 0.15])}
    evaluator.calculate_error_sig_eff_gh = lambda: 0.22
    evaluator.calculate_error_energy_estimate_bdt_reg_tree = lambda: (
        0.33,
        [0.1, 0.2],
        [0.01, 0.02],
    )
    evaluator.calculate_error_gamma_ray_psf = lambda: 0.43
    evaluator.calculate_error_image_template_methods = lambda: 0.53

    evaluator.calculate_metrics()

    assert evaluator.error_eff_area["relative_errors"] == pytest.approx(
        np.array([0.1, 0.2, 0.15]), rel=1e-2
    )
    assert evaluator.error_sig_eff_gh == pytest.approx(0.22, rel=1e-2)
    assert evaluator.error_energy_estimate_bdt_reg_tree == pytest.approx(0.33, rel=1e-2)
    assert evaluator.error_gamma_ray_psf == pytest.approx(0.43, rel=1e-2)
    assert evaluator.error_image_template_methods == pytest.approx(0.53, rel=1e-2)

    expected_results = {
        "error_eff_area": evaluator.error_eff_area,
        "error_sig_eff_gh": evaluator.error_sig_eff_gh,
        "error_energy_estimate_bdt_reg_tree": evaluator.error_energy_estimate_bdt_reg_tree,
        "error_gamma_ray_psf": evaluator.error_gamma_ray_psf,
        "error_image_template_methods": evaluator.error_image_template_methods,
    }
    assert evaluator.metric_results == expected_results


@pytest.fixture
def setup_evaluator():
    file_path = "path_to_fits_file"
    file_type = "On-source"
    metrics = {
        "error_eff_area": 0.05,
        "error_sig_eff_gh": 0.02,
        "error_energy_estimate_bdt_reg_tree": 0.03,
        "error_gamma_ray_psf": 0.01,
        "error_image_template_methods": 0.04,
    }
    grid_point = (1.0, 45.0, 30.0, 0.1, 0.05)

    evaluator = StatisticalErrorEvaluator(file_path, file_type, metrics, grid_point)

    evaluator.metric_results = {
        "error_eff_area": {"relative_errors": np.array([0.04, 0.05, 0.06])},
        "error_sig_eff_gh": 0.02,
        "error_energy_estimate_bdt_reg_tree": 0.03,
        "error_gamma_ray_psf": 0.01,
        "error_image_template_methods": 0.04,
    }

    return evaluator


def test_calculate_overall_metric_average(setup_evaluator):
    evaluator = setup_evaluator
    result = evaluator.calculate_overall_metric(metric="average")

    expected_overall_average = np.mean([0.06, 0.02, 0.03, 0.01, 0.04])
    assert result == pytest.approx(expected_overall_average)


def test_calculate_overall_metric_maximum(setup_evaluator):
    evaluator = setup_evaluator
    result = evaluator.calculate_overall_metric(metric="maximum")

    # Check the overall maximum error value
    expected_overall_max = max([0.06, 0.02, 0.03, 0.01, 0.04])
    assert result == pytest.approx(expected_overall_max)


def test_calculate_overall_metric_invalid_metric(setup_evaluator):
    evaluator = setup_evaluator
    with pytest.raises(ValueError, match="Unsupported metric"):
        evaluator.calculate_overall_metric(metric="invalid_metric")
