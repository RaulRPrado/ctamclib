#!/usr/bin/python3

import logging
import pytest

import numpy as np
import astropy.units as u
from astropy.io import ascii

from simtools import visualize
import simtools.io_handler as io
from simtools import db_handler
import simtools.config as cfg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


def test_plot_1D(db):

    logger.debug("Testing plot1D")

    xTitle = "Wavelength [nm]"
    yTitle = "Mirror reflectivity [%]"
    headersType = {"names": (xTitle, yTitle), "formats": ("f8", "f8")}
    title = "Test 1D plot"

    testFileName = "ref_200_1100_190211a.dat"
    db.getFileDB(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        dest=io.getTestModelDirectory(),
        fileName=testFileName
    )
    testDataFile = cfg.findFile(
        testFileName,
        io.getTestModelDirectory()
    )
    dataIn = np.loadtxt(testDataFile, usecols=(0, 1), dtype=headersType)

    # Change y-axis to percent
    if "%" in yTitle:
        if np.max(dataIn[yTitle]) <= 1:
            dataIn[yTitle] = 100 * dataIn[yTitle]
    data = dict()
    data["Reflectivity"] = dataIn
    for i in range(5):
        newData = np.copy(dataIn)
        newData[yTitle] = newData[yTitle] * (1 - 0.1 * (i + 1))
        data["{}%% reflectivity".format(100 * (1 - 0.1 * (i + 1)))] = newData

    plt = visualize.plot1D(data, title=title, palette="autumn")

    plotFile = io.getTestPlotFile("plot_1D.pdf")
    if plotFile.exists():
        plotFile.unlink()
    plt.savefig(plotFile)

    logger.debug("Produced 1D plot ({}).".format(plotFile))

    assert plotFile.exists()


def test_plot_table(db):

    logger.debug("Testing plotTable")

    title = "Test plot table"

    testFileName = "Transmission_Spectrum_PlexiGlass.dat"
    db.getFileDB(
        dbName="test-data",
        dest=io.getTestModelDirectory(),
        fileName=testFileName
    )
    tableFile = cfg.findFile(
        testFileName,
        io.getTestModelDirectory()
    )
    table = ascii.read(tableFile)

    plt = visualize.plotTable(table, yTitle="Transmission", title=title, noMarkers=True)

    plotFile = io.getTestPlotFile("plot_table.pdf")
    if plotFile.exists():
        plotFile.unlink()
    plt.savefig(plotFile)

    logger.debug("Produced 1D plot ({}).".format(plotFile))

    assert plotFile.exists()


def test_add_unit():

    valueWithUnit = [30, 40] << u.nm
    assert visualize._addUnit("Wavelength", valueWithUnit) == "Wavelength [nm]"
    valueWithoutUnit = [30, 40]
    assert visualize._addUnit("Wavelength", valueWithoutUnit) == "Wavelength"
