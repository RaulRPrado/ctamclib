#!/usr/bin/python3
r"""
    Submit a model parameter value and corresponding metadata through the command line.

    Input and metadata is validated, and if necessary enriched and converted following
    the model parameter schemas.

    Command line arguments
    ----------------------
    parameter (str)
        model parameter name
    value (str, value)
        input value
    instrument (str)
        instrument name.
    site (str)
        site location.
    model_version (str)
        Model version.
    input_meta (str, optional)
        input meta data file (yml format)

    Example
    -------

    Submit the number of gains for the LSTN-design readout chain:

    .. code-block:: console

        simtools-submit-model-parameter-from-external \
            --parameter num_gains \\
            --value 2 \\
            --instrument LSTN-design \\
            --site North \\
            --model_version 6.0.0 \\
            --input_meta num_gains.metadata.yml

"""

import logging
from pathlib import Path

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model.metadata_collector import MetadataCollector


def _parse(label, description):
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
        "--parameter", type=str, required=True, help="Parameter for simulation model"
    )
    config.parser.add_argument("--instrument", type=str, required=True, help="Instrument name")
    config.parser.add_argument("--site", type=str, required=True, help="Site location")
    config.parser.add_argument("--model_version", type=str, required=True, help="Model version")

    config.parser.add_argument("--value", type=str, required=True, help="Model parameter value")

    config.parser.add_argument(
        "--input_meta",
        help="meta data file associated to input data",
        type=str,
        required=False,
    )
    return config.initialize(output=True)


def main():  # noqa: D103
    args_dict, _ = _parse(
        label=Path(__file__).stem,
        description="Submit and validate a model parameters).",
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    output_file = (
        args_dict.get("output_file")
        if args_dict.get("output_file")
        else f"{args_dict['parameter']}.json"
    )

    writer.ModelDataWriter.dump_model_parameter(
        parameter_name=args_dict["parameter"],
        value=args_dict["value"],
        instrument=args_dict["instrument"],
        model_version=args_dict["model_version"],
        output_file=output_file,
        use_plain_output_path=args_dict.get("use_plain_output_path"),
        metadata=MetadataCollector(args_dict=args_dict).top_level_meta,
    )


if __name__ == "__main__":
    main()
