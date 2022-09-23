#!/usr/bin/python3

import logging

import astropy.units as u
import pytest
from astropy.table import Table

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.camera_efficiency import CameraEfficiency
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


@pytest.fixture
def telescope_model(set_db):
    telescopeModel = TelescopeModel(
        site="North",
        telescopeModelName="LST-1",
        modelVersion="Prod5",
        label="validate_camera_efficiency",
    )
    return telescopeModel


@pytest.fixture
def camera_efficiency(telescope_model):
    camera_efficiency = CameraEfficiency(telescopeModel=telescope_model, test=True)
    return camera_efficiency


@pytest.fixture
def results_file(db):
    testFileName = "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv"
    db.exportFileDB(
        dbName="test-data",
        dest=io.getOutputDirectory(
            filesLocation=cfg.get("outputLocation"),
            label="validate_camera_efficiency",
            dirType="camera-efficiency",
            test=True,
        ),
        fileName=testFileName,
    )

    return io.getOutputDirectory(
        filesLocation=cfg.get("outputLocation"),
        label="validate_camera_efficiency",
        dirType="camera-efficiency",
        test=True,
    ).joinpath("camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv")


def test_from_kwargs(telescope_model):

    telModel = telescope_model
    label = "test-from-kwargs"
    zenithAngle = 30 * u.deg
    ce = CameraEfficiency.fromKwargs(
        telescopeModel=telModel, label=label, zenithAngle=zenithAngle, test=True
    )
    assert ce.config.zenithAngle == 30


def test_validate_telescope_model(cfg_setup):

    with pytest.raises(ValueError):
        CameraEfficiency(
            telescopeModel="bla_bla",
        )


def test_load_files(camera_efficiency):
    assert (
        camera_efficiency._fileResults.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv"
    )
    assert (
        camera_efficiency._fileSimtel.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.dat"
    )
    assert (
        camera_efficiency._fileLog.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.log"
    )


def test_read_results(camera_efficiency, results_file):
    camera_efficiency._readResults()
    assert isinstance(camera_efficiency._results, Table)
    assert camera_efficiency._hasResults is True


def test_calc_camera_efficiency(telescope_model, camera_efficiency, results_file):
    camera_efficiency._readResults()
    telescope_model.exportModelFiles()
    assert camera_efficiency.calcCameraEfficiency() == pytest.approx(
        0.24468117923810984
    )  # Value for Prod5 LST-1


def test_calc_tel_efficiency(telescope_model, camera_efficiency, results_file):
    camera_efficiency._readResults()
    telescope_model.exportModelFiles()
    assert camera_efficiency.calcTelEfficiency() == pytest.approx(
        0.23988884493787524
    )  # Value for Prod5 LST-1


def test_calc_tot_efficiency(telescope_model, camera_efficiency, results_file):
    camera_efficiency._readResults()
    telescope_model.exportModelFiles()
    assert camera_efficiency.calcTotEfficiency(
        camera_efficiency.calcTelEfficiency()
    ) == pytest.approx(
        0.48018680628175714
    )  # Value for Prod5 LST-1


def test_calc_reflectivity(camera_efficiency, results_file):
    camera_efficiency._readResults()
    assert camera_efficiency.calcReflectivity() == pytest.approx(
        0.9167918392938349
    )  # Value for Prod5 LST-1


def test_calc_nsb_rate(telescope_model, camera_efficiency, results_file):
    camera_efficiency._readResults()
    telescope_model.exportModelFiles()
    assert camera_efficiency.calcNsbRate()[0] == pytest.approx(
        0.24421390533203186
    )  # Value for Prod5 LST-1
