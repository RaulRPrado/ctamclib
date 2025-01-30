#!/usr/bin/python3

r"""
Generate markdown reports for relevant model parameters.

The generated reports include detailed information on each parameter,
such as the parameter name, value, unit, description, and short description.
The reports are then uploaded as GitLab Pages using GitLab's CI workflow.
"""

import logging
import textwrap
from itertools import groupby
from operator import itemgetter
from pathlib import Path

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
        db_config=True, simulation_model=["site", "telescope", "model_version"]
    )


def generate_markdown_report(output_path, args_dict, data):
    """
    Generate a markdown file to report the parameter values.

    Parameters
    ----------
    output_path : str
        The folder where the markdown file will be saved.
    args_dict : dict
        Configuration arguments including model version and telescope name.
    data : list
        Parameter data in the format:
        [class, parameter_name, value, description, short_description]
    """
    # Sort data by class to prepare for grouping
    data.sort(key=itemgetter(0, 1))  # Sort by class and alphabetically

    output_filename = output_path / Path(args_dict["telescope"] + ".md")

    # Start writing the Markdown file
    with Path(output_filename).open("w", encoding="utf-8") as file:
        # Group by class and write sections
        file.write(f"# {args_dict['telescope']}\n")
        for class_name, group in groupby(data, key=itemgetter(0)):
            file.write(f"## {class_name}\n\n")

            # Write table header and separator row
            file.write("| Parameter Name      | Values      | Short Description           |\n")
            file.write("|---------------------|-------------|-----------------------------|\n")

            # Write table rows
            column_widths = [25, 25, 80]
            for _, parameter_name, value, description, short_description in group:
                text = short_description if short_description else description
                wrapped_text = textwrap.fill(str(text), column_widths[2]).split("\n")
                wrapped_text = " ".join(wrapped_text)
                file.write(
                    f"| {parameter_name:{column_widths[0]}} |"
                    f" {value:{column_widths[1]}} |"
                    f" {wrapped_text} |\n"
                )
            file.write("\n\n")


def main():  # noqa: D103
    label_name = Path(__file__).stem
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

    parameter_data = ReadParameters(
        telescope_model,
        output_path,
    ).get_telescope_parameter_data()

    generate_markdown_report(output_path, args, parameter_data)

    logger.info(
        f"Markdown report generated for {args['site']}"
        f" Telescope {args['telescope']} (v{args['model_version']})"
    )

    if args["parameter"]:
        logger.info(
            f"Comparing parameters across model versions for Telescope: {args['telescope']}"
            f"and Site: {args['site']}."
        )

        output_path = io_handler_instance.get_output_directory(
            label=label_name, sub_dir=f"parameters/{args['telescope']}"
        )

        read_params = ReadParameters(telescope_model, output_path)

        all_params = read_params.get_all_parameter_descriptions()[0]

        for parameter in all_params:
            if telescope_model.has_parameter(parameter):
                comparison_data = read_params.compare_parameter_across_versions(
                    parameter, telescope_model
                )
                if comparison_data:
                    output_filename = output_path / f"{parameter}.md"
                    with output_filename.open("w", encoding="utf-8") as file:
                        # Write header
                        file.write(f"# {parameter}\n\n")
                        file.write(f"**Telescope**: {args['telescope']}\n\n")
                        file.write(f"**Description**: {comparison_data[0]['description']}\n\n")
                        file.write("\n")

                        # Write table header
                        file.write("| Model Version      | Value                |\n")
                        file.write("|--------------------|----------------------|\n")

                        # Write table rows
                        for item in comparison_data:
                            file.write(f"| {item['model_version']} | {item['value']} |\n")

                        file.write("\n")
                        if isinstance(comparison_data[0]["value"], str) and comparison_data[0][
                            "value"
                        ].endswith(".md)"):
                            file.write(
                                f"![Parameter plot.](../../_images/"
                                f"{args['telescope']}_{parameter}.png)"
                            )


if __name__ == "__main__":
    main()
