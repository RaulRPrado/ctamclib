#!/usr/bin/python3

"""
    Summary
    -------
    This application validates the optical model parameters through ray tracing simulations \
    of the whole telescope, assuming a point-like light source. The output includes PSF (D80), \
    effective mirror area and effective focal length as a function of the off-axis angle. \

    The telescope zenith angle and the source distance can be set by command line arguments.

    Examples of the plots generated by this applications are shown below. On the top, the D80 \
    vs off-axis is shown in cm (left) and deg (right). On the bottom, the effective mirror \
    area (left) and the effective focal length (right) vs off-axis angle are shown.

    .. _validate_optics_plot:
    .. image::  images/validate_optics_North-LST-1_d80_cm.png
      :width: 49 %
    .. image::  images/validate_optics_North-LST-1_d80_deg.png
      :width: 49 %

    .. image::  images/validate_optics_North-LST-1_eff_area.png
      :width: 49 %
    .. image::  images/validate_optics_North-LST-1_eff_flen.png
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
    max_offset (float, optional)
        Maximum offset angle in deg (default=4).
    test (activation mode, optional)
        If activated, application will be faster by simulating fewer photons.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    LST-1 Prod5

    Runtime about 1-2 min.

    .. code-block:: console

        python applications/validate_optics.py --site North --telescope LST-1 --max_offset 5.0

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
"""

import logging
import matplotlib.pyplot as plt
import argparse
import numpy as np

import astropy.units as u

import simtools.config as cfg
import simtools.util.general as gen
import simtools.io_handler as io
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing

# from simtools.visualize import setStyle

# setStyle()


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
        "--max_offset",
        help="Maximum offset angle in deg (default=4)",
        type=float,
        default=4,
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
    label = "validate_optics"

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    # Output directory to save files related directly to this app
    outputDir = io.getApplicationOutputDirectory(cfg.get("outputLocation"), label)

    telModel = TelescopeModel(
        site=args.site,
        telescopeModelName=args.telescope,
        modelVersion=args.model_version,
        label=label,
        readFromDB=True,
    )

    print(
        "\nValidating telescope optics with ray tracing simulations"
        " for {}\n".format(telModel.name)
    )

    ray = RayTracing.fromKwargs(
        telescopeModel=telModel,
        sourceDistance=args.src_distance * u.km,
        zenithAngle=args.zenith * u.deg,
        offAxisAngle=np.linspace(0, args.max_offset, int(args.max_offset / 0.25) + 1)
        * u.deg,
    )
    ray.simulate(test=args.test, force=False)
    ray.analyze(force=True)

    # Plotting
    for key in ["d80_deg", "d80_cm", "eff_area", "eff_flen"]:
        plt.figure(figsize=(8, 6), tight_layout=True)

        ray.plot(key, marker="o", linestyle=":", color="k")

        plotFileName = label + "_" + telModel.name + "_" + key
        plotFile = outputDir.joinpath(plotFileName)
        plt.savefig(str(plotFile) + ".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(str(plotFile) + ".png", format="png", bbox_inches="tight")
        plt.clf()
