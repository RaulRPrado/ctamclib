#!/usr/bin/python3

"""
    Summary
    -------
    This application simulates showers to be used in trigger rate calculations.
    Arrays with one (1MST) or four telescopes (4LST) can be used, in case of \
    mono or stereo trigger configurations, respectively.

    Simulations are managed by the shower_simulator module.
    Each run is simulated in a job. Each job is submitted by using the submission \
    command from the global config settings (see config_template.yml). \
    The config entry extra_commands can be used to extra commands to be ran in each job,
    before the actual simulation.

    At the moment, the shower simulations are performed by CORSIKA, which requires \
    the zstd package. Please, make sure that the command to set your zstd path is \
    properly set by the extra_commands in the command line configuration.

    Command line arguments
    ----------------------
    array (str, required)
        Name of the array (1MST, 4LST ...).
    site (str, required)
        South or North.
    primary (str, required)
        Name of the primary particle (proton, helium ...).
    nruns (int, optional)
        Number of runs to be simulated (default=100).
    nevents (int, optional)
        Number of events simulated per run (default=100000).
    zenith (float, optional)
        Zenith angle in deg (default=20).
    azimuth (float, optional)
        Azimuth angle in deg (default=0).
    output (str, optional)
        Path of the directory to store the output simulations. By default, \
        the standard output directory defined by config will be used.
    test (activation mode, optional)
        If activated, no job will be submitted. Instead, an example of the \
        run script will be printed.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    Producing a set of proton showers for trigger rate simulations of LST.

    .. code-block:: console

        python applications/sim_showers_for_trigger_rates.py -a 4LST -s North \
        --primary proton --nruns 100 --nevents 10000 --output {some dir for large files}
"""

import logging
from pathlib import Path

import astropy.units as u

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools import io_handler
from simtools.simulator import Simulator


def _parse(label=None, description=None):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        application label.
    description: str
        description of application.

    Returns
    -------
    CommandLineParser
        command line parser object

    """

    config = configurator.Configurator(label=label, description=description)
    config.parser.add_argument(
        "--array",
        help="Name of the array (e.g. 1MST, 4LST ...)",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--primary",
        help="Name of the primary particle (e.g. proton, helium ...)",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--nruns", help="Number of runs (default=100)", type=int, default=100
    )
    config.parser.add_argument(
        "--nevents", help="Number of events/run (default=100)", type=int, default=100000
    )
    config.parser.add_argument(
        "--zenith", help="Zenith angle in deg (default=20)", type=float, default=20
    )
    config.parser.add_argument(
        "--azimuth", help="Azimuth angle in deg (default=0)", type=float, default=0
    )
    # TODO confusing with output_path?
    config.parser.add_argument(
        "--output",
        help="Path of the output directory where the simulations will be saved.",
        type=str,
        default=None,
    )
    return config.initialize(telescope_model=True, job_submission=True)


def main():

    label = Path(__file__).stem
    args_dict, _ = _parse(
        label=label, description="Simulate showers to be used for trigger rate calculations"
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label, dir_type="application-plots")

    shower_config_data = {
        "data_directory": args_dict["output"],
        "site": args_dict["site"],
        "layout_name": args_dict["array"],
        "run_range": [1, args_dict["nruns"] + 1],
        "nshow": args_dict["nevents"],
        "primary": args_dict["primary"],
        "erange": [10 * u.GeV, 300 * u.TeV],
        "eslope": -2,
        "zenith": args_dict["zenith"] * u.deg,
        "azimuth": args_dict["azimuth"] * u.deg,
        "viewcone": 10 * u.deg,
        "cscat": [20, 1500 * u.m, 0],
    }

    shower_simulator = Simulator(
        label=label,
        simulator="corsika",
        simulator_source_path=args_dict.get("simtelpath", None),
        config_data=shower_config_data,
        submit_command=args_dict.get("submit_command", ""),
        test=args_dict["test"],
    )

    if not args_dict["test"]:
        shower_simulator.simulate()
    else:
        logger.info("Test flag is on - it will not submit any job.")
        logger.info("This is an example of the run script:")
        shower_simulator.simulate()

    # Exporting the list of output/log/input files into the application folder
    output_file_list = output_dir.joinpath("output_files_{}.list".format(args_dict["primary"]))
    log_file_list = output_dir.joinpath("log_files_{}.list".format(args_dict["primary"]))

    def print_list_into_file(list_of_files, file_name):
        with open(file_name, "w") as f:
            for line in list_of_files:
                f.write(line + "\n")

    logger.info("List of output files exported to {}".format(output_file_list))
    print_list_into_file(shower_simulator.get_list_of_output_files(), output_file_list)
    logger.info("List of log files exported to {}".format(log_file_list))
    print_list_into_file(shower_simulator.get_list_of_log_files(), log_file_list)


if __name__ == "__main__":
    main()
