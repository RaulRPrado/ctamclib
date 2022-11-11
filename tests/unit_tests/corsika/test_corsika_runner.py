#!/usr/bin/python3

import logging
import shutil

import astropy.units as u
import pytest

import simtools.util.general as gen
from simtools.corsika.corsika_runner import CorsikaRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def corsika_config_data():
    return {
        "data_directory": ".",
        "nshow": 10,
        "primary": "gamma",
        "erange": [100 * u.GeV, 1 * u.TeV],
        "eslope": -2,
        "zenith": 20 * u.deg,
        "azimuth": 0 * u.deg,
        "viewcone": 0 * u.deg,
        "cscat": [10, 1500 * u.m, 0],
    }


@pytest.fixture
def corsika_runner(corsika_config_data, io_handler, simtelpath):

    corsika_runner = CorsikaRunner(
        site="south",
        layout_name="test-layout",
        simtel_source_path=simtelpath,
        label="test-corsika-runner",
        corsika_config_data=corsika_config_data,
    )
    return corsika_runner


@pytest.fixture
def corsika_file(io_handler):
    corsika_file = io_handler.get_input_data_file(
        file_name="run1_proton_za20deg_azm0deg_North_1LST_test-lst-array.corsika.zst", test=True
    )
    return corsika_file


def test_get_run_script(corsika_runner):
    # No run number is given

    script = corsika_runner.get_run_script()

    assert script.exists()

    # Run number is given
    run_number = 3
    script = corsika_runner.get_run_script(run_number=run_number)

    assert script.exists()


def test_get_run_script_with_invalid_run(corsika_runner):
    for run in [-2, "test"]:
        with pytest.raises(ValueError):
            _ = corsika_runner.get_run_script(run_number=run)


def test_run_script_with_extra(corsika_runner):

    extra = ["testing", "testing-extra-2"]
    script = corsika_runner.get_run_script(run_number=3, extra_commands=extra)

    assert gen.file_has_text(script, "testing-extra-2")


def test_get_info_for_file_name(corsika_runner):
    info_for_file_name = corsika_runner.get_info_for_file_name(run_number=1)
    assert info_for_file_name["run"] == 1
    assert info_for_file_name["primary"] == "gamma"
    assert info_for_file_name["array_name"] == "TestLayout"
    assert info_for_file_name["site"] == "South"
    assert info_for_file_name["label"] == "test-corsika-runner"


def test_get_file_name(corsika_runner, io_handler):
    info_for_file_name = corsika_runner.get_info_for_file_name(run_number=1)
    file_name = "corsika_run1_gamma_South_TestLayout_test-corsika-runner"

    assert corsika_runner.get_file_name(
        "log", **info_for_file_name
    ) == corsika_runner._corsika_log_dir.joinpath(f"log_{file_name}.log")

    assert corsika_runner.get_file_name(
        "corsika_log", **info_for_file_name
    ) == corsika_runner._corsika_data_dir.joinpath(corsika_runner._get_run_directory(1)).joinpath(
        "run1.log"
    )

    script_file_dir = io_handler.get_output_directory("test-corsika-runner", "corsika").joinpath(
        "scripts"
    )
    assert corsika_runner.get_file_name("script", **info_for_file_name) == script_file_dir.joinpath(
        f"{file_name}.sh"
    )

    file_name_for_output = (
        "corsika_run1_gamma_za20deg_azm0deg_South_TestLayout_test-corsika-runner.zst"
    )
    assert corsika_runner.get_file_name(
        "output", **info_for_file_name
    ) == corsika_runner._corsika_data_dir.joinpath(corsika_runner._get_run_directory(1)).joinpath(
        file_name_for_output
    )

    sub_log_file_dir = io_handler.get_output_directory("test-corsika-runner", "corsika").joinpath(
        "logs"
    )
    assert corsika_runner.get_file_name(
        "sub_log", **info_for_file_name, mode="out"
    ) == sub_log_file_dir.joinpath(f"log_sub_{file_name}.out")
    with pytest.raises(ValueError):
        corsika_runner.get_file_name("foobar", **info_for_file_name, mode="out")


def test_has_file(corsika_runner, corsika_file):
    # Copying the corsika file to the expected location and
    # changing its name for the sake of this test.
    # This should not affect the efficacy of this test.
    output_directory = corsika_runner._corsika_data_dir.joinpath(
        corsika_runner._get_run_directory(1)
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        corsika_file,
        output_directory.joinpath(
            "corsika_run1_gamma_za20deg_azm0deg_South_TestLayout_test-corsika-runner.zst"
        ),
    )
    assert corsika_runner.has_file(file_type="output", run_number=1)
    assert not corsika_runner.has_file(file_type="log", run_number=1234)
