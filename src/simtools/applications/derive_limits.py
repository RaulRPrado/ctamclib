#!/usr/bin/python3

r"""
Derives the limits for energy, radial distance, and viewcone to be used in CORSIKA simulations.

The limits are derived based on the event loss fraction specified by the user.

Command line arguments
----------------------
event_data_file (str, required)
    Path to the file containing the event data.
loss_fraction (float, required)
    Fraction of events to be lost.


Example
-------
Derive limits for a given file with a specified loss fraction.

.. code-block:: console

    simtools-derive-limits\\
        --event_data_file path/to/event_data_file.hdf5 \\
        --loss_fraction 1e-6
"""

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations.hdf5_handler import read_hdf5
from simtools.production_configuration.limits_calculation import LimitCalculator

_logger = logging.getLogger(__name__)


def _parse():
    """
    Parse command line configuration.

    Parameters
    ----------
    event_data_file: str
        The event data file.
    loss_fraction: float
        Loss fraction of events for limit derivation.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(
        description="Derive limits for energy, radial distance, and viewcone."
    )
    config.parser.add_argument(
        "--event_data_file",
        type=str,
        required=True,
        help="Path to the event data file containing the event data.",
    )
    config.parser.add_argument(
        "--loss_fraction", type=float, required=True, help="Fraction of events to be lost."
    )
    return config.initialize(db_config=False)


def main():
    """Derive limits for energy, radial distance, and viewcone."""
    args_dict, _ = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    event_data_file_path = args_dict["event_data_file"]
    loss_fraction = args_dict["loss_fraction"]

    _logger.info(f"Loading event data file: {event_data_file_path}")
    tables = read_hdf5(event_data_file_path)

    calculator = LimitCalculator(tables)

    lower_energy_limit = calculator.compute_lower_energy_limit(loss_fraction)
    _logger.info(f"Lower energy threshold: {lower_energy_limit}")

    upper_radial_distance = calculator.compute_upper_radial_distance(loss_fraction)
    _logger.info(f"Upper radius threshold: {upper_radial_distance}")

    viewcone = calculator.compute_viewcone(loss_fraction)
    _logger.info(f"Viewcone radius: {viewcone}")


if __name__ == "__main__":
    main()
