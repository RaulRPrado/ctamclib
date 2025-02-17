#!/usr/bin/python3

r"""
Generate markdown reports for relevant model parameters.

The generated reports include detailed information on each parameter,
such as the parameter name, value, unit, description, and short description.
The reports are then uploaded as GitLab Pages using GitLab's CI workflow.
"""

import logging

from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.reporting.docs_read_parameters import ReadParameters
from simtools.utils import general as gen


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=("Generate a markdown report for model parameters."),
    )

    config.parser.add_argument(
        "--parameter",
        action="store_true",
        help="Compare all parameters across model versions for one telescope.",
    )

    return config.initialize(
        db_config=True, simulation_model=["site", "telescope", "model_version", "parameter_version"]
    )


def main():  # noqa: D103
    label_name = "reports"
    args, db_config = _parse(label_name)
    io_handler_instance = io_handler.IOHandler()
    output_path = io_handler_instance.get_output_directory(
        label=label_name, sub_dir=f"productions/{args['model_version']}"
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args["log_level"]))

    telescope_model = TelescopeModel(
        site=args["site"],
        telescope_name=args["telescope"],
        model_version=args["model_version"],
        label=label_name,
        mongo_db_config=db_config,
    )

    ReadParameters(
        db_config,
        telescope_model,
        output_path,
    ).generate_array_element_report()

    logger.info(
        f"Markdown report generated for {args['site']}"
        f" Telescope {args['telescope']} (v{args['model_version']}):"
        f" {output_path}"
    )


if __name__ == "__main__":
    main()
