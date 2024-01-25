#!/usr/bin/python3

"""
    Summary
    -------
    This application calculates the number of events for a proton distribution within a given
    angular cone and writes the result to the console.

    Command line arguments
    ----------------------
    inner (float, optional)
        Inner radius of the angular cone in radians.
    outer (float, required)
        Outer radius of the angular cone in radians.
    obs_time (float, required)
        Observation time in seconds.
    area (float, required)
        Observation area in square meters.
    energy_min (float, required)
        Minimum energy for integration in TeV.
    energy_max (float, required)
        Maximum energy for integration in TeV.

    Example
    -------
    .. code-block:: console

        calculate_proton_events --inner 0 --outer 0.1745 --obs_time 1 --area 12 --energy_min 0.008
        --energy_max 10
"""

import logging
from pathlib import Path

import astropy.units as u
from ctao_cosmic_ray_spectra.spectral import (
    IRFDOC_PROTON_SPECTRUM as ProtonDistribution,
)

import simtools.utils.general as gen
from simtools.configuration import configurator


def _parse(label, description):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing the application.
    description: str
        Description of the application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--inner",
        help="Inner radius of the angular cone in radians.",
        type=float,
        required=False,
        default=0,
    )

    config.parser.add_argument(
        "--outer",
        help="Outer radius of the angular cone in radians.",
        type=float,
        required=True,
    )

    config.parser.add_argument(
        "--obs_time",
        help="Observation time in seconds.",
        type=float,
        required=True,
    )

    config.parser.add_argument(
        "--area",
        help="Observation area in square meters.",
        type=float,
        required=True,
    )

    config.parser.add_argument(
        "--energy_min",
        help="Minimum energy for integration in TeV.",
        type=float,
        required=True,
    )

    config.parser.add_argument(
        "--energy_max",
        help="Maximum energy for integration in TeV.",
        type=float,
        required=True,
    )

    config_parser, _ = config.initialize(db_config=False, paths=True)
    return config_parser


def main():
    label = Path(__file__).stem
    description = "Calculate the number of proton events within a given angular cone."
    config_parser = _parse(label, description)
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(config_parser["log_level"]))
    logger.info("Starting the application.")

    inner = config_parser["inner"] * u.rad
    outer = config_parser["outer"] * u.rad
    obs_time = config_parser["obs_time"] * u.s
    area = config_parser["area"] * u.m**2
    energy = (config_parser["energy_min"], config_parser["energy_max"]) * u.TeV

    logger.info(f"Calculating proton events for angular cone: inner={inner}, outer={outer}")
    logger.info(f"Observation time: {obs_time}, Observation area: {area}")
    logger.info(f"Energy range for integration: {energy}")

    final = ProtonDistribution.derive_number_events(  # pylint: disable=E1101
        inner, outer, obs_time, area, energy  # pylint: disable=E1101
    )  # pylint: disable=E1101
    # 12: E1101: Instance of 'PowerLaw' has no 'derive_number_events' member (no-member)

    logger.info(f"Number of proton events within the specified cone: {final}")


if __name__ == "__main__":
    main()
