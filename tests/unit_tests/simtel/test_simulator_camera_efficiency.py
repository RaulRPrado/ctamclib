#!/usr/bin/python3

import logging
from pathlib import Path

import astropy.units as u
import pytest

from simtools.camera.camera_efficiency import CameraEfficiency
from simtools.simtel.simulator_camera_efficiency import SimulatorCameraEfficiency

logger = logging.getLogger()


@pytest.fixture
def simulator_camera_efficiency(camera_efficiency_sst, simtel_path):
    camera_efficiency_sst.export_model_files()
    return SimulatorCameraEfficiency(
        simtel_path=simtel_path,
        telescope_model=camera_efficiency_sst.telescope_model,
        file_simtel=camera_efficiency_sst._file["simtel"],
        label="test-simtel-runner-camera-efficiency",
    )


@pytest.fixture
def expected_command():
    return [
        "testeff",
        "-fnsb",
        "alt 2147.0 -fatm atm_trans_2147_1_10_2_0_2147.dat",
        "-flen 2.15191 -spix 0.6",
        "weighted_average_1D_ref_astri-2d_2018-01-17.dat -m2",
        "-teltrans 0.921",
        "transmission_sstcam_weighted_220512.dat",
        "-fqe PDE_lvr3_6mm_75um_uncoated_5.9V.dat",
    ]


@pytest.fixture
def benn_ellison_spectrum_file_name():
    return "Benn_LaPalma_sky_converted.lis"


def test_make_run_command(
    simulator_camera_efficiency, expected_command, benn_ellison_spectrum_file_name
):
    command, std_out_file, std_err_file = simulator_camera_efficiency._make_run_command()

    for item in expected_command:
        assert item in command

    assert "-nc" not in command
    # Benn_LaPalma_sky_converted.lis is the default nsb spectrum
    assert benn_ellison_spectrum_file_name in command

    assert isinstance(std_out_file, Path)
    assert std_err_file is None


def test_make_run_command_with_nsb_spectrum(simulator_camera_efficiency, expected_command):
    simulator_camera_efficiency.nsb_spectrum = (
        "tests/resources/benn_ellison_spectrum_for_testing.txt"
    )
    command, _, _ = simulator_camera_efficiency._make_run_command()

    for item in expected_command:
        assert item in command

    assert "benn_ellison_spectrum_for_testing.txt" in command


def test_make_run_command_without_altitude_correction(
    simulator_camera_efficiency, expected_command, benn_ellison_spectrum_file_name
):
    simulator_camera_efficiency.skip_correction_to_nsb_spectrum = True
    command, _, _ = simulator_camera_efficiency._make_run_command()

    for item in expected_command:
        assert item in command

    assert "-nc" in command
    # Benn_LaPalma_sky_converted.lis is the default nsb spectrum
    assert benn_ellison_spectrum_file_name in command


def test_check_run_result(simulator_camera_efficiency):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """

    with pytest.raises(RuntimeError):
        simulator_camera_efficiency._check_run_result()


@pytest.mark.xfail(reason="Test requires Derived-Values Database")
def test_get_one_dim_distribution(io_handler, db_config, simtel_path, model_version_prod5):

    logger.warning(
        "Running test_get_one_dim_distribution using prod5 model "
        " (prod6 model with 1D transmission function)"
    )
    camera_efficiency_sst_prod5 = CameraEfficiency(
        config_data={
            "telescope": "SSTS-design",
            "site": "South",
            "model_version": model_version_prod5,
            "zenith_angle": 20 * u.deg,
            "azimuth_angle": 0 * u.deg,
        },
        db_config=db_config,
        simtel_path=simtel_path,
        label="validate_camera_efficiency",
        test=True,
    )

    # 2D transmission window not defined in prod6; required prod5 runner
    camera_efficiency_sst_prod5.export_model_files()
    simulator_camera_efficiency_prod5 = SimulatorCameraEfficiency(
        simtel_path=simtel_path,
        telescope_model=camera_efficiency_sst_prod5.telescope_model,
        file_simtel=camera_efficiency_sst_prod5._file["simtel"],
        label="test-simtel-runner-camera-efficiency",
    )
    camera_filter_file = simulator_camera_efficiency_prod5._get_one_dim_distribution(
        "camera_filter", "camera_filter_incidence_angle"
    )
    assert camera_filter_file.exists()


def test_validate_or_fix_nsb_spectrum_file_format(simulator_camera_efficiency):
    """
    Test that the function returns a file with the correct format.
    The test is run twice, once on a file with the wrong format and then
    the produced file is tested as well in order to make sure that
    the function does not change the format of a file with the correct format.
    """
    validated_nsb_spectrum_file = (
        simulator_camera_efficiency._validate_or_fix_nsb_spectrum_file_format(
            "tests/resources/benn_ellison_spectrum_for_testing.txt"
        )
    )
    assert validated_nsb_spectrum_file.exists()

    def produced_file_has_expected_values(file):
        # Test that the first 3 non-comment lines are the following values:
        wavelengths = [300.00, 315.00, 330.00]
        nsbs = [0, 0.612, 1.95]
        with open(file, encoding="utf-8") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                entry = line.split()
                assert float(entry[0]) == pytest.approx(wavelengths.pop(0))
                assert float(entry[2]) == pytest.approx(nsbs.pop(0))
                assert len(entry) == 3
                if len(wavelengths) == 0:
                    break

    produced_file_has_expected_values(validated_nsb_spectrum_file)
    # Test that the function does not change the format of a file with the correct format
    second_validated_nsb_spectrum_file = (
        simulator_camera_efficiency._validate_or_fix_nsb_spectrum_file_format(
            validated_nsb_spectrum_file
        )
    )
    produced_file_has_expected_values(second_validated_nsb_spectrum_file)
