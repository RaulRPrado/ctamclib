#!/usr/bin/python3
r"""
    Convert all simulation model parameters exported from sim_telarray format using schema files.

    Check value, type, and range, convert units, and write json files
    ready to be submitted to the model database. Prints out parameters which are not found
    in simtel configuration file and parameters which are not found in simtools schema files.

    Note that all parameters are assigned the same parameter version.

    Command line arguments
    ----------------------
    simtel_cfg_file (str)
        File name of sim_telarray configuration file containing all simulation model parameters.

    simtel_telescope_name (str)
        Name of the telescope in the sim_telarray configuration file.

    telescope (str, optional)
        Telescope model name (e.g. LST-1, SST-D, ...)

    Example
    -------

    Extract model parameters with schema files from simtel configuration file
    (requires access to the model parameter repository)

    .. code-block:: console

       simtools-convert-all-model-parameters-from-simtel \\
          --schema_directory ../model_parameters/schema\\
          --simtel_cfg_file all_telescope_config_la_palma.cfg\\
          --simtel_telescope_name CT1\\
          --telescope LSTN-01\\
          --parameter_version "1.0.0"

    The export of the model parameters from sim_telarray for 6.0.0 can be done e.g., as follows:

    .. code-block:: console

        ./sim_telarray/bin/sim_telarray -c sim_telarray/cfg/CTA/CTA-PROD6-LaPalma.cfg \\
            -C limits=no-internal -C initlist=no-internal -C list=no-internal \\
            -C typelist=no-internal -C maximum_telescopes=30 -DNSB_AUTOSCALE \\
            -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=30 /dev/null \\
            2>|/dev/null | grep '(@cfg)'  >| all_telescope_config_la_palma.cfg

        ./sim_telarray/bin/sim_telarray -c sim_telarray/cfg/CTA/CTA-PROD6-Paranal.cfg \\
            -C limits=no-internal -C initlist=no-internal -C list=no-internal \\
            -C typelist=no-internal -C maximum_telescopes=87 -DNSB_AUTOSCALE \\
            -DFLASHCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=87 /dev/null \\
            2>|/dev/null | grep '(@cfg)'  >| all_telescope_config_paranal.cfg

"""

import logging
import re
from pathlib import Path

import numpy as np

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model import schema
from simtools.io_operations.io_handler import IOHandler
from simtools.simtel.simtel_config_reader import SimtelConfigReader


def _parse(label=None, description=None):
    """
    Parse command line configuration.

    Parameters
    ----------
    label: str
        Label describing application.
    description: str
        Description of application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--schema_directory",
        help=(
            "Directory with schema files for model parameter validation "
            "(default: simtools schema directory)"
        ),
        required=False,
    )
    config.parser.add_argument(
        "--simtel_cfg_file",
        help="File name for simtel_array configuration",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--simtel_telescope_name",
        help="Name of the telescope in the sim_telarray configuration file",
        type=str,
        required=True,
    )
    return config.initialize(simulation_model=["telescope", "parameter_version"])


def get_list_of_simtel_parameters(simtel_config_file, logger):
    """
    Return list of simtel parameters found in simtel configuration file.

    Parameters
    ----------
    simtel_config_file: str
        File name for sim_telarray configuration
    logger: logging.Logger
        Logger object

    Returns
    -------
    list
        List of parameters found in simtel configuration file.

    """
    simtel_parameter_set = set()
    with open(simtel_config_file, encoding="utf-8") as file:
        for line in file:
            parts_of_lines = re.split(r",\s*|\s+", line.strip())
            simtel_parameter_set.add(parts_of_lines[1].lower())
    logger.info(f"Found {len(simtel_parameter_set)} parameters in simtel configuration file.")
    return list(simtel_parameter_set)


def read_simtel_config_file(args_dict, logger, schema_file, camera_pixels=None):
    """
    Read the simtel configuration file.

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object
    schema_file: str
        Schema path name.
    camera_pixels: int
        Number of camera pixels.

    """
    simtel_config_reader = SimtelConfigReader(
        schema_file=schema_file,
        simtel_config_file=args_dict["simtel_cfg_file"],
        simtel_telescope_name=args_dict["simtel_telescope_name"],
        camera_pixels=camera_pixels,
    )
    logger.info(f"Simtel parameter: {simtel_config_reader.parameter_dict}")

    if simtel_config_reader.parameter_dict is None or len(simtel_config_reader.parameter_dict) == 0:
        return None
    return simtel_config_reader


def get_number_of_camera_pixel(args_dict, logger):
    """
    Get the number of camera pixels from the simtel configuration file.

    Required to set the dimension some of the parameter correctly, as simtel
    in some cases does not provide the dimension ('all:' in the parameter files).

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object

    Returns
    -------
    int, None
        Number of camera pixels (None if file is not found)

    """
    try:
        simtel_config_reader = SimtelConfigReader(
            schema_file=schema.model_parameter_schema_file("camera_pixels"),
            simtel_config_file=args_dict["simtel_cfg_file"],
            simtel_telescope_name=args_dict["simtel_telescope_name"],
        )
        _camera_pixel = simtel_config_reader.parameter_dict.get(args_dict["simtel_telescope_name"])
    except (FileNotFoundError, AttributeError):
        logger.warning("Failed to read camera pixel parameter.")
        _camera_pixel = None
    logger.info(f"Number of camera pixels: {_camera_pixel}")
    return _camera_pixel


def read_and_export_parameters(args_dict, logger):
    """
    Read and export parameters from simtel configuration file to json files.

    Provide extensive logging information on the parameters found in the simtel
    configuration file.

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object

    Returns
    -------
    list
        List of simtel parameters not found in schema files.
    list
        List of simtools parameter not found in simtel configuration file.

    """
    _parameters, _schema_files = schema.model_parameter_schema_files(
        args_dict.get("schema_directory")
    )
    _simtel_parameters = get_list_of_simtel_parameters(args_dict["simtel_cfg_file"], logger)

    io_handler = IOHandler()
    io_handler.set_paths(output_path=args_dict["output_path"], use_plain_output_path=True)

    _camera_pixel = get_number_of_camera_pixel(args_dict, logger)

    _parameters_not_in_simtel = []

    for _parameter, _schema_file in zip(_parameters, _schema_files):

        logger.info(f"Parameter: {_parameter} Schema file: {_schema_file}")
        simtel_config_reader = read_simtel_config_file(
            args_dict, logger, _schema_file, _camera_pixel
        )

        if simtel_config_reader is None:
            logger.info("Parameter not found in sim_telarray configuration file.")
            _parameters_not_in_simtel.append(_parameter)
            continue

        _json_dict = writer.ModelDataWriter.dump_model_parameter(
            parameter_name=_parameter,
            value=simtel_config_reader.parameter_dict.get(args_dict["simtel_telescope_name"]),
            instrument=args_dict["telescope"],
            parameter_version=args_dict["parameter_version"],
            output_file=io_handler.get_output_file(f"{_parameter}.json"),
        )

        simtel_config_reader.compare_simtel_config_with_schema()

        if simtel_config_reader.simtel_parameter_name.lower() in _simtel_parameters:
            _simtel_parameters.remove(simtel_config_reader.simtel_parameter_name.lower())

        if _json_dict["file"]:
            logger.info(f"File name for {_parameter} is {_json_dict['value']}")

    return _parameters_not_in_simtel, _simtel_parameters


def print_parameters_not_found(_parameters_not_in_simtel, _simtel_parameters, args_dict, logger):
    """
    Print simtel/simtools parameter not found in schema and ocnfiguration files.

    For simtel parameters not found, check if the setting for the chose
    telescope is different from the default values.

    Parameters
    ----------
    _parameters_not_in_simtel: list
        List of simtel parameters not found in schema files.
    _simtel_parameters: list
        List of simtel parameters not found in simtools schema files.
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object

    """
    logger.info(
        f"Parameters not found in simtools schema files ({len(_parameters_not_in_simtel)}):"
    )
    for para in sorted(_parameters_not_in_simtel):
        logger.info(f"  {para}")

    logger.info(f"Simtel parameters not found in schema files ({len(_simtel_parameters)}):")
    for para in sorted(_simtel_parameters):
        logger.info(f"Simtel parameter: {para}")
        simtel_config_reader = SimtelConfigReader(
            schema_file=None,
            simtel_config_file=args_dict["simtel_cfg_file"],
            simtel_telescope_name=args_dict["simtel_telescope_name"],
            parameter_name=para,
        )
        _default = simtel_config_reader.parameter_dict.get("default")
        _tel_value = simtel_config_reader.parameter_dict.get(args_dict["simtel_telescope_name"])
        # simple comparison of default value and telescope values, does not work for lists
        try:
            if _default == _tel_value or np.isclose(_default, _tel_value):
                logger.info(f"    Default and telescope values are equal: {_default}")
                continue
        except (ValueError, TypeError):
            pass
        if isinstance(_default, np.ndarray):
            logger.warning(f"    Default value: {_default} (length: {_default.size})")
        else:
            logger.warning(f"    Default value: {_default}")
        if isinstance(_tel_value, np.ndarray):
            logger.warning(f"    Telescope value: {_tel_value} (length: {_tel_value.size})")
        else:
            logger.warning(f"    Telescope value: {_tel_value}")


def print_list_of_files(args_dict, logger):
    """
    Print model parameters which describe a file name.

    This is useful to find files which are part of the model.

    Parameters
    ----------
    args_dict: dict
        Dictionary with command line arguments.
    logger: logging.Logger
        Logger object

    """
    model_files = sorted(Path(args_dict["output_path"]).rglob("*.json"))
    for file in model_files:
        model_dict = gen.collect_data_from_file(file_name=file)
        if model_dict.get("file"):
            logger.info(f"{file.name}: {model_dict['value']}")


def main():  # noqa: D103
    args_dict, _ = _parse(
        label=Path(__file__).stem,
        description="Convert simulation model parameters from sim_telarray to simtools format.",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _parameters_not_in_simtel, _simtel_parameters = read_and_export_parameters(args_dict, logger)
    print_parameters_not_found(_parameters_not_in_simtel, _simtel_parameters, args_dict, logger)
    print_list_of_files(args_dict, logger)


if __name__ == "__main__":
    main()
