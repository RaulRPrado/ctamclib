#!/usr/bin/python3

import logging

import astropy.units as u
import numpy as np
import pytest

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.layout.layout_array import LayoutArray
from simtools.layout.telescope_position import MissingInputForConvertion

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def layoutCenterDataDict():
    return {
        "center_lon": -17.8920302 * u.deg,
        "center_lat": 28.7621661 * u.deg,
        "center_easting": None,
        "center_northing": None,
        "EPSG": 32628,
        "center_alt": 2177 * u.m,
    }


@pytest.fixture
def corsikaTelescopeDataDict():
    return {
        "corsika_sphere_radius": {"LST": 12.5 * u.m, "MST": 9.6 * u.m, "SST": 3 * u.m},
        "corsika_sphere_center": {"LST": 16 * u.m, "MST": 9 * u.m, "SST": 3.25 * u.m},
        "corsika_obs_level": 2158 * u.m,
    }


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


@pytest.fixture
def telescopeTestFile(db):
    testFileName = "telescope_positions_prod5_north.ecsv"
    db.exportFileDB(dbName="test-data", dest=io.getTestModelDirectory(), fileName=testFileName)

    cfgFile = cfg.findFile(testFileName, io.getTestModelDirectory())
    return cfgFile


def test_fromLayoutArrayName(cfg_setup):

    layout = LayoutArray.fromLayoutArrayName("south-Prod5")

    assert 99 == layout.getNumberOfTelescopes()


def test_initalizeCoordinateSystems(layoutCenterDataDict):

    layout = LayoutArray(name="testLayout")
    with pytest.raises(MissingInputForConvertion):
        layout._initalizeCoordinateSystems(None, False)

    layout._initalizeCoordinateSystems(None, True)
    _x, _y, _z = layout._arrayCenter.getCoordinates("corsika")
    assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == 0.0 * u.m
    _lat, _lon, _z = layout._arrayCenter.getCoordinates("mercator")
    assert np.isnan(_lat) and np.isnan(_lon)

    layout._initalizeCoordinateSystems(layoutCenterDataDict, False)
    _x, _y, _z = layout._arrayCenter.getCoordinates("corsika")
    assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == layoutCenterDataDict["center_alt"]
    _lat, _lon, _z = layout._arrayCenter.getCoordinates("mercator")
    assert _lat.value == pytest.approx(layoutCenterDataDict["center_lat"].value, 1.0e-2)
    assert _lon.value == pytest.approx(layoutCenterDataDict["center_lon"].value, 1.0e-2)
    _E, _N, _z = layout._arrayCenter.getCoordinates("utm")
    assert _E.value == pytest.approx(217609.0, 1.0)
    assert _N.value == pytest.approx(3185067.0, 1.0)


def test_read_tel_list(cfg_setup, telescopeTestFile):

    layout = LayoutArray(name="testLayout")
    layout.readTelescopeListFile(telescopeTestFile)
    layout.convertCoordinates()
    assert 19 == layout.getNumberOfTelescopes()


def test_add_tel(cfg_setup, telescopeTestFile):

    layout = LayoutArray(name="testLayout")
    layout.readTelescopeListFile(telescopeTestFile)
    ntel_before = layout.getNumberOfTelescopes()
    layout.addTelescope("L-05", "corsika", 100.0 * u.m, 50.0 * u.m, 2177.0 * u.m)
    ntel_after = layout.getNumberOfTelescopes()
    assert ntel_before + 1 == ntel_after

    layout.addTelescope("L-05", "corsika", 100.0 * u.m, 50.0 * u.m, None, 50.0 * u.m)
    assert layout._telescopeList[-1].getAltitude().value == pytest.approx(2192.0)


def test_build_layout(cfg_setup, layoutCenterDataDict, corsikaTelescopeDataDict):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layoutCenterData=layoutCenterDataDict,
        corsikaTelescopeData=corsikaTelescopeDataDict,
    )

    layout.addTelescope(
        telescopeName="L-01", crsName="corsika", xx=57.5 * u.m, yy=57.5 * u.m, telCorsikaZ=0 * u.m
    )
    layout.addTelescope(
        telescopeName="L-02", crsName="corsika", xx=-57.5 * u.m, yy=57.5 * u.m, telCorsikaZ=0 * u.m
    )
    layout.addTelescope(
        telescopeName="L-02", crsName="corsika", xx=57.5 * u.m, yy=-57.5 * u.m, telCorsikaZ=0 * u.m
    )
    layout.addTelescope(
        telescopeName="L-04", crsName="corsika", xx=-57.5 * u.m, yy=-57.5 * u.m, telCorsikaZ=0 * u.m
    )

    layout.convertCoordinates()
    layout.printTelescopeList()
    layout.exportTelescopeList("corsika")

    # Building a second layout from the file exported by the first one
    layout_2 = LayoutArray("test_layout_2")
    layout_2.readTelescopeListFile(layout.telescopeListFile)

    assert 4 == layout_2.getNumberOfTelescopes()
    assert layout_2._arrayCenter.getAltitude().value == pytest.approx(
        layout._arrayCenter.getAltitude().value, 1.0e-2
    )


def test_converting_center_coordinates(cfg_setup, layoutCenterDataDict, corsikaTelescopeDataDict):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layoutCenterData=layoutCenterDataDict,
        corsikaTelescopeData=corsikaTelescopeDataDict,
    )

    _lat, _lon, _ = layout._arrayCenter.getCoordinates("mercator")
    assert _lat.value == pytest.approx(28.7621661)
    assert _lon.value == pytest.approx(-17.8920302)

    _east, _north, _ = layout._arrayCenter.getCoordinates("utm")
    assert _north.value == pytest.approx(3185067.28)
    assert _east.value == pytest.approx(217609.23)

    assert layout._arrayCenter.getAltitude().value == pytest.approx(2177.0)


def test_getCorsikaInputList(layoutCenterDataDict, corsikaTelescopeDataDict, telescopeTestFile):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layoutCenterData=layoutCenterDataDict,
        corsikaTelescopeData=corsikaTelescopeDataDict,
    )
    layout.addTelescope(
        telescopeName="L-01", crsName="corsika", xx=57.5 * u.m, yy=57.5 * u.m, telCorsikaZ=0 * u.m
    )
    corsikaInputList = layout.getCorsikaInputList()

    assert corsikaInputList == "TELESCOPE\t 57.500E2\t 57.500E2\t 0.000E2\t 12.500E2\t # L-01\n"
