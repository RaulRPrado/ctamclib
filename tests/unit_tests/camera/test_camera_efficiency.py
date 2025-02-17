#!/usr/bin/python3

import logging
import shutil

import astropy.units as u
import pytest
from astropy.table import Table

from simtools.camera.camera_efficiency import CameraEfficiency
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simulator_camera_efficiency import SimulatorCameraEfficiency

logger = logging.getLogger()


@pytest.fixture
def config_data_lst(model_version_prod5):
    return {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": model_version_prod5,
        "zenith_angle": 20 * u.deg,
        "azimuth_angle": 0 * u.deg,
    }


@pytest.fixture
def camera_efficiency_lst(io_handler, db_config, simtel_path, config_data_lst):
    return CameraEfficiency(
        config_data=config_data_lst,
        db_config=db_config,
        simtel_path=simtel_path,
        label="validate_camera_efficiency",
        test=True,
    )


@pytest.fixture
def prepare_results_file(io_handler):
    test_file_name = (
        "tests/resources/"
        "camera-efficiency-table-North-LSTN-01-za20.0deg_azm000deg_validate_camera_efficiency.ecsv"
    )
    output_directory = io_handler.get_output_directory(
        label="validate_camera_efficiency",
        sub_dir="camera-efficiency",
    )
    shutil.copy(test_file_name, output_directory)
    return output_directory.joinpath(test_file_name)


def test_report(camera_efficiency_lst):
    assert str(camera_efficiency_lst) == "CameraEfficiency(label=validate_camera_efficiency)\n"


def test_initialize_simulation_models(db_config, config_data_lst, camera_efficiency_lst):
    tel_model, site_model = camera_efficiency_lst._initialize_simulation_models(
        config_data_lst, db_config
    )
    assert isinstance(tel_model, TelescopeModel)
    assert isinstance(site_model, SiteModel)


def test_configuration_from_args_dict(camera_efficiency_lst):
    _config = camera_efficiency_lst._configuration_from_args_dict(
        {
            "zenith_angle": 30 * u.deg,
            "azimuth_angle": 90 * u.deg,
            "nsb_spectrum": "dark",
        }
    )
    assert isinstance(_config, dict)
    assert pytest.approx(_config["zenith_angle"]) == 30.0
    assert pytest.approx(_config["azimuth_angle"]) == 90.0
    assert _config["nsb_spectrum"] == "dark"

    _config_none = camera_efficiency_lst._configuration_from_args_dict(
        {
            "zenith_angle": None,
            "azimuth_angle": None,
            "nsb_spectrum": None,
        }
    )
    assert pytest.approx(_config_none["zenith_angle"]) == 20.0
    assert pytest.approx(_config_none["azimuth_angle"]) == 0.0
    assert _config_none["nsb_spectrum"] is None


def test_load_files(camera_efficiency_lst):
    _name = "camera-efficiency-table-North-LSTN-01-za20.0deg_azm000deg_validate_camera_efficiency"
    assert camera_efficiency_lst._file["results"].name == _name + ".ecsv"
    _name = "camera-efficiency-North-LSTN-01-za20.0deg_azm000deg_validate_camera_efficiency"
    assert camera_efficiency_lst._file["simtel"].name == _name + ".dat"
    assert camera_efficiency_lst._file["log"].name == _name + ".log"


def test_simulate(io_handler, camera_efficiency_lst, caplog, mocker):

    mock_run = mocker.patch.object(SimulatorCameraEfficiency, "run")
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst.simulate()
        assert "Simulating CameraEfficiency" in caplog.text
    mock_run.assert_called_once()


def test_read_results(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    assert isinstance(camera_efficiency_lst._results, Table)
    assert camera_efficiency_lst._has_results is True


def test_calc_camera_efficiency(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.export_model_files()
    assert camera_efficiency_lst.calc_camera_efficiency() == pytest.approx(
        0.24468117923810984
    )  # Value for Prod5 LST-1


def test_calc_tel_efficiency(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.export_model_files()
    assert camera_efficiency_lst.calc_tel_efficiency() == pytest.approx(
        0.23988884493787524
    )  # Value for Prod5 LST-1


def test_calc_tot_efficiency(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.export_model_files()
    assert camera_efficiency_lst.calc_tot_efficiency(
        camera_efficiency_lst.calc_tel_efficiency()
    ) == pytest.approx(
        0.48018680628175714
    )  # Value for Prod5 LST-1


def test_calc_reflectivity(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    assert camera_efficiency_lst.calc_reflectivity() == pytest.approx(
        0.9167918392938349
    )  # Value for Prod5 LST-1


def test_calc_nsb_rate(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.export_model_files()
    _, nsb_rate_ref_conditions = camera_efficiency_lst.calc_nsb_rate()
    assert nsb_rate_ref_conditions == pytest.approx(0.24421390533203186)  # Value for Prod5 LST-1


def test_export_results(mocker, camera_efficiency_lst, caplog, prepare_results_file):
    # no results available yet
    with caplog.at_level(logging.ERROR):
        camera_efficiency_lst.export_results()
    assert "Cannot export results because they do not exist" in caplog.text

    # results available
    mocker.patch.object(camera_efficiency_lst, "results_summary", return_value="TestString")
    camera_efficiency_lst._read_results()
    mock_file = mocker.mock_open()
    mocker.patch("builtins.open", mock_file)
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst.export_results()
    assert "Exporting summary results" in caplog.text
    mock_file().write.assert_called_once_with("TestString")


def test_analyze_has_results(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.analyze()
    assert camera_efficiency_lst._has_results is True


def test_analyze_from_file(camera_efficiency_lst, mocker):
    camera_efficiency_lst._file["simtel"] = (
        "tests/resources/"
        "camera-efficiency-North-MSTx-NectarCam-za20.0deg_azm000deg_validate_camera_efficiency.dat"
    )
    mocker.patch.object(CameraEfficiency, "results_summary", return_value="summary")
    camera_efficiency_lst.analyze(export=False, force=True)
    assert camera_efficiency_lst._has_results is True
    assert isinstance(camera_efficiency_lst._results, Table)
    assert len(camera_efficiency_lst._results) > 1
    assert "N4" in camera_efficiency_lst._results.colnames


def test_results_summary(camera_efficiency_lst, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.export_model_files()
    summary = camera_efficiency_lst.results_summary()
    assert "Results summary for LSTN-01" in summary


def test_plot_efficiency(camera_efficiency_lst, mocker, prepare_results_file):
    camera_efficiency_lst._read_results()
    camera_efficiency_lst.export_model_files()
    plot_table_mock = mocker.patch("simtools.visualization.visualize.plot_table")
    camera_efficiency_lst.plot_efficiency(efficiency_type="NSB")
    plot_table_mock.assert_called_once()


def test_save_plot(camera_efficiency_lst, mocker, caplog):
    fig_mock = mocker.MagicMock()
    with caplog.at_level(logging.INFO):
        camera_efficiency_lst._save_plot(fig_mock, "test_plot")
    assert "Saved plot test_plot efficiency to" in caplog.text
