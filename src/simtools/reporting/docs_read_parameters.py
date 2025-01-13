#!/usr/bin/python3

r"""Class to read and manage relevant model parameters for a given telescope model."""

import logging
import os
from pathlib import Path

from simtools.utils import names


class ReadParameters:
    """Read and manage model parameter data."""

    def __init__(self, telescope_model, output_path):
        """Initialise class with a telescope model."""
        self._logger = logging.getLogger(__name__)
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
        output_data_path = Path(self.output_path, "data_files")
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_file = output_data_path / Path(input_file.stem + ".md")

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

        return output_file

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

    def get_telescope_parameter_data(self):
        """
        Get model parameter data.

        Returns
        -------
            list: A list of lists containing parameter names, values with units,
                  descriptions, and short descriptions.
        """
        parameter_descriptions = self.get_all_parameter_descriptions()
        self.telescope_model.export_model_files()

        data = []

        for parameter in parameter_descriptions[0]:
            if not self.telescope_model.has_parameter(parameter):
                continue
            value = self.telescope_model.get_parameter_value_with_unit(parameter)
            if self.telescope_model.get_parameter_file_flag(parameter) and value:
                try:
                    input_file_name = self.telescope_model.config_file_directory / Path(value)
                    output_file_name = self._convert_to_md(input_file_name)
                    value = f"[{os.path.basename(value)}]({output_file_name.as_posix()})"
                except FileNotFoundError:
                    value = f"File not found: {value}"
            elif isinstance(value, list):
                value = ", ".join(str(q) for q in value)
            else:
                value = str(value)

            description = parameter_descriptions[0].get(parameter)
            short_description = parameter_descriptions[1].get(parameter, description)
            inst_class = parameter_descriptions[2].get(parameter)
            data.append([inst_class, parameter, value, description, short_description])

        return data

    def compare_parameter_across_versions(self, parameter_name, telescope_model):
        """
        Compare a parameter's value across different model versions.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to compare.
        telescope_model : TelescopeModel
            The telescope model instance.

        Returns
        -------
        list
            A list of dictionaries containing model version, parameter value, and description.
        """
        all_versions = ["5.0.0", "6.0.0"]  # TODO: remove hard coding of version list
        comparison_data = []

        for version in all_versions:
            telescope_model.model_version = version
            parameter_data = self.get_telescope_parameter_data()

            for param in parameter_data:
                if param[1] == parameter_name:  # Match the parameter name
                    comparison_data.append(
                        {
                            "model_version": version,
                            "value": param[2],
                            "description": param[3],
                        }
                    )
                    break

        return comparison_data
