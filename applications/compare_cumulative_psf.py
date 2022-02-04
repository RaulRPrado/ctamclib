#!/usr/bin/python3

"""
    Summary
    -------
    This application simulates the cumulative PSF and compare with data (if available).

    The telescope zenith angle and the source distance can be set by command line arguments.

    The measured cumulative PSF should be provided by using the command line argument data. \
    A file name is expected, in which the file should contains 3 columns: radial distance in mm, \
    differential value of photon intensisity and its integral value.

    The MC model can be changed by providing a yaml file with the new parameter values using \
    the argument pars (see example below).

    Examples of the plots generated by this applications are shown below. On the left, \
    the cumulative PSF and on the right, the simulated PSF image.

    .. _compare_cumulative_psf_plot:
    .. image::  images/compare_cumulative_psf_North-LST-1_cumulativePSF.png
      :width: 49 %
    .. image::  images/compare_cumulative_psf_North-LST-1_image.png
      :width: 49 %

    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LST-1, SST-D, ...).
    model_version (str, optional)
        Model version (default=prod4).
    src_distance (float, optional)
        Source distance in km (default=10).
    zenith (float, optional)
        Zenith angle in deg (default=20).
    data (str, optional)
        Name of the data file with the measured cumulative PSF.
    pars (str, optional)
        Yaml file with the new model parameters to replace the default ones.
    test (activation mode, optional)
        If activated, application will be faster by simulating fewer photons.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    LST-1 Prod5

    Runtime < 1 min.

    First, create an yml file named lst_pars.yml with the following content:

    .. code-block:: yaml

        mirror_reflection_random_angle: '0.0075,0.15,0.035'
        mirror_align_random_horizontal: '0.0040,28.,0.0,0.0'
        mirror_align_random_vertical: '0.0040,28.,0.0,0.0'

    And the run:

    .. code-block:: console

        python applications/compare_cumulative_psf.py --site North --telescope LST-1 --model_version prod4 --pars lst_pars.yml --data PSFcurve_data_v2.txt

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
"""

import logging
import matplotlib.pyplot as plt
import argparse
import yaml
from collections import OrderedDict

import numpy as np
import astropy.units as u

import simtools.io_handler as io
import simtools.util.general as gen
import simtools.config as cfg
from simtools.ray_tracing import RayTracing
from simtools.model.telescope_model import TelescopeModel
from simtools import visualize


def loadData(datafile):
    dType = {"names": ("Radius [cm]", "Relative intensity"), "formats": ("f8", "f8")}
    # testDataFile = io.getTestDataFile('PSFcurve_data_v2.txt')
    data = np.loadtxt(datafile, dtype=dType, usecols=(0, 2))
    data["Radius [cm]"] *= 0.1
    data["Relative intensity"] /= np.max(np.abs(data["Relative intensity"]))
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Calculate and plot the PSF and eff. mirror area as a function of off-axis angle "
            "of the telescope requested."
        )
    )
    parser.add_argument("-s", "--site", help="North or South", type=str, required=True)
    parser.add_argument(
        "-t",
        "--telescope",
        help="Telescope model name (e.g. MST-FlashCam-D, LST-1)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model_version",
        help="Model version (default=prod4)",
        type=str,
        default="prod4",
    )
    parser.add_argument(
        "--src_distance",
        help="Source distance in km (default=10)",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--zenith", help="Zenith angle in deg (default=20)", type=float, default=20
    )
    parser.add_argument(
        "--data", help="Data file name with the measured PSF vs radius [cm]", type=str
    )
    parser.add_argument(
        "--pars", help="Yaml file with the model parameters to be replaced", type=str
    )
    parser.add_argument(
        "--test",
        help="Test option will be faster by simulating fewer photons.",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        dest="logLevel",
        action="store",
        default="info",
        help="Log level to print (default is INFO)",
    )

    args = parser.parse_args()
    label = "compare_cumulative_psf"

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    # Output directory to save files related directly to this app
    outputDir = io.getApplicationOutputDirectory(cfg.get("outputLocation"), label)

    telModel = TelescopeModel(
        site=args.site,
        telescopeModelName=args.telescope,
        modelVersion=args.model_version,
        label=label,
    )

    # New parameters
    if args.pars is not None:
        with open(args.pars) as file:
            newPars = yaml.load(file, Loader=yaml.FullLoader)
        telModel.changeMultipleParameters(**newPars)

    ray = RayTracing.fromKwargs(
        telescopeModel=telModel,
        sourceDistance=args.src_distance * u.km,
        zenithAngle=args.zenith * u.deg,
        offAxisAngle=[0.0 * u.deg],
    )

    ray.simulate(test=args.test, force=False)
    ray.analyze(force=False)

    # Plotting cumulative PSF
    im = ray.images()[0]

    print("d80 in cm = {}".format(im.getPSF()))

    # Plotting cumulative PSF
    dataToPlot = OrderedDict()
    dataToPlot[r"sim$\_$telarray"] = im.getCumulativeData()
    if args.data is not None:
        dataFile = cfg.findFile(args.data)
        dataToPlot['measured'] = loadData(dataFile)
    fig = visualize.plot1D(dataToPlot)
    fig.gca().set_ylim(0, 1.05)

    plotFileName = label + "_" + telModel.name + "_cumulativePSF"
    plotFile = outputDir.joinpath(plotFileName)
    for f in ["pdf", "png"]:
        plt.savefig(str(plotFile) + "." + f, format=f, bbox_inches="tight")
    fig.clf()

    # Plotting image
    dataToPlot = im.getImageData()
    visualize.plotHist2D(dataToPlot, bins=80)
    circle = plt.Circle((0, 0), im.getPSF(0.8) / 2, color="k", fill=False, lw=2, ls="--")
    fig.gca().add_artist(circle)

    plotFileName = label + "_" + telModel.name + "_image"
    plotFile = outputDir.joinpath(plotFileName)
    for f in ["pdf", "png"]:
        fig.savefig(str(plotFile) + "." + f, format=f, bbox_inches="tight")
    fig.clf()