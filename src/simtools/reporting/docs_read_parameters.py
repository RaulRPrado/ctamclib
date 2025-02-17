#!/usr/bin/python3

r"""Class to read and manage relevant model parameters for a given telescope model."""

import logging
import textwrap
from itertools import groupby
from pathlib import Path

from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.utils import names

logger = logging.getLogger()


class ReadParameters:
    """Read and manage model parameter data."""

    def __init__(self, db_config, telescope_model, output_path):
        """Initialise class with a telescope model."""
        self._logger = logging.getLogger(__name__)
        self.db_config = db_config
        self.telescope_model = telescope_model
        self.output_path = output_path

    def _convert_to_md(self, input_file):
        """
        Convert a '.dat' or '.ecsv' file to a Markdown file, preserving formatting.

        Parameters
        ----------
        input_file: Path, str
           Simulation data file (in '.dat' or '.ecsv' format).

        Returns
        -------
        - Path to the created Markdown file.
        """
        input_file = Path(input_file)
        output_data_path = Path(self.output_path / "_data_files")
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_file_name = Path(input_file.stem + ".md")
        output_file = output_data_path / output_file_name

        with (
            input_file.open("r", encoding="utf-8") as infile,
            output_file.open("w", encoding="utf-8") as outfile,
        ):

            outfile.write(f"# {input_file.stem}")
            outfile.write("\n")
            outfile.write("```")
            outfile.write("\n")
            file_contents = infile.read()
            outfile.write(file_contents)
            outfile.write("\n")
            outfile.write("```")

        return f"_data_files/{output_file_name}"

    def get_all_parameter_descriptions(self):
        """
        Get descriptions for all model parameters.

        Returns
        -------
            tuple: A tuple containing two dictionaries:
                - parameter_description: Maps parameter names to their descriptions.
                - short_description: Maps parameter names to their short descriptions.
                - inst_class: Maps parameter names to their respective class.
        """
        parameter_description, short_description, inst_class = {}, {}, {}

        for instrument_class in names.instrument_classes("telescope"):
            for parameter, details in names.load_model_parameters(instrument_class).items():
                parameter_description[parameter] = details.get("description")
                short_description[parameter] = details.get("short_description")
                inst_class[parameter] = instrument_class

        return parameter_description, short_description, inst_class

    def get_telescope_parameter_data(self, telescope_model):
        """
        Get model parameter data.

        Parameters
        ----------
        telescope_model : TelescopeModel
            The telescope model instance.

        Returns
        -------
            list: A list of lists containing parameter names, values with units,
                  descriptions, and short descriptions.
        """
        all_params = telescope_model.db.get_model_parameters(
            site=telescope_model.site,
            array_element_name=telescope_model.name,
            collection="telescopes",
        )

        telescope_model.export_model_files()
        parameter_descriptions = self.get_all_parameter_descriptions()
        data = []

        if not any(
            all_params[parameter]["instrument"] == telescope_model.name for parameter in all_params
        ):
            data = "No telescope-specific parameters, check telescope design report."
            logger.info({data})
            return data

        for parameter in all_params:
            if not telescope_model.has_parameter(parameter):
                continue

            if all_params[parameter]["instrument"] == telescope_model.name:
                parameter_version = telescope_model.get_parameter_version(parameter)
                value = telescope_model.get_parameter_value_with_unit(parameter)
                if telescope_model.get_parameter_file_flag(parameter) and value:
                    try:
                        input_file_name = telescope_model.config_file_directory / Path(value)
                        output_file_name = self._convert_to_md(input_file_name)
                        value = f"[{Path(value).name}]({output_file_name})"
                    except FileNotFoundError:
                        value = f"File not found: {value}"
                elif isinstance(value, list):
                    value = ", ".join(str(q) for q in value)
                else:
                    value = str(value)

                description = parameter_descriptions[0].get(parameter)
                short_description = parameter_descriptions[1].get(parameter, description)
                inst_class = parameter_descriptions[2].get(parameter)
                data.append(
                    [
                        inst_class,
                        parameter,
                        parameter_version,
                        value,
                        description,
                        short_description,
                    ]
                )

        return data

    def _compare_parameter_across_versions(self, parameter_name):
        """
        Compare a parameter's value across different model versions.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to compare.

        Returns
        -------
        list
            A list of dictionaries containing model version, parameter value, and description.
        """
        all_versions = self.telescope_model.db.get_model_versions()
        all_versions.reverse()
        comparison_data = []

        for model_version in all_versions:
            telescope_model = TelescopeModel(
                site=self.telescope_model.site,
                telescope_name=self.telescope_model.name,
                model_version=model_version,
                label="reports",
                mongo_db_config=self.db_config,
            )

            if not telescope_model.has_parameter(parameter_name):
                return comparison_data

            parameter_data = self.get_telescope_parameter_data(telescope_model)
            for param in parameter_data:
                if param[1] == parameter_name:
                    comparison_data.append(
                        {
                            "model_version": model_version,
                            "parameter_version": param[2],
                            "value": param[3],
                            "description": param[4],
                        }
                    )
                    break
        return comparison_data

    def generate_array_element_report(self):
        """
        Generate a markdown report of all model parameters per array element.

        Output
        ----------
        One markdown report of a given array element listing parameter values,
        versions, and descriptions.
        """
        output_filename = Path(self.output_path / (self.telescope_model.name + ".md"))

        output_filename.parent.mkdir(parents=True, exist_ok=True)

        data = self.get_telescope_parameter_data(self.telescope_model)
        # Sort data by class to prepare for grouping
        if not isinstance(data, str):
            data.sort(key=lambda x: (x[0], x[1]), reverse=True)

        with output_filename.open("w", encoding="utf-8") as file:
            # Group by class and write sections
            file.write(f"# {self.telescope_model.name}\n")

            if isinstance(data, str):
                file.write(data)
                return

            for class_name, group in groupby(data, key=lambda x: x[0]):
                group = sorted(group, key=lambda x: x[1])
                file.write(f"## {class_name}\n\n")

                # Write table header and separator row
                file.write(
                    "| Parameter Name      |  Parameter Version     "
                    "| Values      | Short Description           |\n"
                    "|---------------------|------------------------"
                    "|-------------|-----------------------------|\n"
                )

                # Write table rows
                column_widths = [20, 20, 20, 70]
                for (
                    _,
                    parameter_name,
                    parameter_version,
                    value,
                    description,
                    short_description,
                ) in group:
                    text = short_description if short_description else description
                    wrapped_text = textwrap.fill(str(text), column_widths[3]).split("\n")
                    wrapped_text = " ".join(wrapped_text)
                    file.write(
                        f"| {parameter_name:{column_widths[0]}} |"
                        f" {parameter_version:{column_widths[1]}} |"
                        f" {value:{column_widths[2]}} |"
                        f" {wrapped_text} |\n"
                    )
                file.write("\n\n")

    def generate_parameter_report(self):
        """
        Generate a markdown report per parameters per array element.

        Output
        ----------
        One markdown report per model parameter of a given array element comparing
        values across model versions.
        """
        logger.info(
            f"Comparing parameters across model versions for Telescope: {self.telescope_model.name}"
            f" and Site: {self.telescope_model.site}."
        )
        io_handler_instance = io_handler.IOHandler()
        output_path = io_handler_instance.get_output_directory(
            label="reports", sub_dir=f"parameters/{self.telescope_model.name}"
        )

        all_params = self.telescope_model.db.get_model_parameters(
            site=self.telescope_model.site,
            array_element_name=self.telescope_model.name,
            collection="telescopes",
        )
        if not any(
            all_params[parameter]["instrument"] == self.telescope_model.name
            for parameter in all_params
        ):
            logger.info("No telescope-specific parameters, check telescope design report.")
            return

        for parameter in all_params:
            comparison_data = []
            if all_params[parameter]["instrument"] == self.telescope_model.name:
                comparison_data = self._compare_parameter_across_versions(parameter)
            if comparison_data:
                output_filename = output_path / f"{parameter}.md"
                with output_filename.open("w", encoding="utf-8") as file:
                    # Write header
                    file.write(
                        f"# {parameter}\n\n"
                        f"**Telescope**: {self.telescope_model.name}\n\n"
                        f"**Description**: {comparison_data[0]['description']}\n\n"
                        "\n"
                    )

                    # Write table header
                    file.write(
                        "| Model Version      | Parameter Version      "
                        "| Value                |\n"
                        "|--------------------|------------------------"
                        "|----------------------|\n"
                    )

                    # Write table rows
                    for item in comparison_data:
                        file.write(
                            f"| {item['model_version']} |"
                            f" {item['parameter_version']} |"
                            f"{item['value'].replace('](', '](../')} |\n"
                        )

                    file.write("\n")
                    if isinstance(comparison_data[0]["value"], str) and comparison_data[0][
                        "value"
                    ].endswith(".md)"):
                        file.write(
                            f"![Parameter plot.](_images/"
                            f"{self.telescope_model.name}_{parameter}.png)"
                        )
