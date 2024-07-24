#!/usr/bin/python3
"""Configuration file writer for sim_telarray."""

import logging

import numpy as np

import simtools.utils.general as gen
import simtools.version
from simtools.utils import names

__all__ = ["SimtelConfigWriter"]


class SimtelConfigWriter:
    """
    SimtelConfigWriter writes sim_telarray configuration files.

    It is designed to be used by model classes (TelescopeModel and ArrayModel) only.

    Parameters
    ----------
    site: str
        South or North.
    model_version: str
        Version of the model (ex. prod5).
    telescope_model_name: str
        Telescope model name.
    layout_name: str
        Layout name.
    label: str
        Instance label. Important for output file naming.
    """

    TAB = " " * 3

    def __init__(
        self, site, model_version, layout_name=None, telescope_model_name=None, label=None
    ):
        """Initialize SimtelConfigWriter."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelConfigWriter")

        self._site = site
        self._model_version = model_version
        self._label = label
        self._layout_name = layout_name
        self._telescope_model_name = telescope_model_name

    def write_telescope_config_file(self, config_file_path, parameters, config_parameters=None):
        """
        Write the sim_telarray config file for a single telescope.

        Parameters
        ----------
        config_file_path: str or Path
            Path of the file to write on.
        parameters: dict
            Model parameters
        config_parameters: dict
            Simulation software configuration parameters
        """
        self._logger.debug(f"Writing telescope config file {config_file_path}")
        with open(config_file_path, "w", encoding="utf-8") as file:
            self._write_header(file, "TELESCOPE CONFIGURATION FILE")

            file.write("#ifdef TELESCOPE\n")
            file.write(
                f"   echo Configuration for {self._telescope_model_name}"
                " - TELESCOPE $(TELESCOPE)\n"
            )
            file.write("#endif\n\n")

            if config_parameters:
                parameters.update(config_parameters)

            for _simtel_name, value in parameters.items():
                if _simtel_name:
                    file.write(f"{_simtel_name} = {self._get_value_string_for_simtel(value)}\n")
            _config_meta = self._get_simtel_metadata("telescope")
            for _simtel_name, value in _config_meta.items():
                file.write(f"{_simtel_name} = {value}\n")

    def _get_value_string_for_simtel(self, value):
        """
        Return a value string for simtel.

        Parameters
        ----------
        value: any
            Value to convert to string.

        Returns
        -------
        str
            Value string for simtel.
        """
        value = "none" if value is None else value  # simtel requires 'none'
        if isinstance(value, bool):
            value = 1 if value else 0
        elif isinstance(value, (list, np.ndarray)):  # noqa: UP038
            value = gen.convert_list_to_string(value)
        return value

    def _get_simtel_metadata(self, config_type):
        """
        Return simtel metadata.

        Parameters
        ----------
        type: str
            Type of the configuration file (telescope, site)

        Returns
        -------
        dict
            Dictionary with simtel metadata.

        """
        parameters = {}
        parameters["config_release"] = (
            f"{self._model_version} written by simtools v{simtools.version.__version__}"
        )
        parameters["config_version"] = self._model_version
        if config_type == "telescope":
            parameters["camera_config_name"] = self._telescope_model_name
            parameters["camera_config_variant"] = ""
            parameters["camera_config_version"] = self._model_version
            parameters["optics_config_name"] = self._telescope_model_name
            parameters["optics_config_variant"] = ""
            parameters["optics_config_version"] = self._model_version
        elif config_type == "site":
            parameters["site_config_name"] = self._site
            parameters["site_config_variant"] = ""
            parameters["site_config_version"] = self._model_version
            parameters["array_config_name"] = self._layout_name
            parameters["array_config_variant"] = ""
            parameters["array_config_version"] = self._model_version
        else:
            raise ValueError(f"Unknown metadata type {config_type}")
        return parameters

    def write_array_config_file(self, config_file_path, telescope_model, site_model):
        """
        Write the sim_telarray config file for an array of telescopes.

        Parameters
        ----------
        config_file_path: str or Path
            Path of the file to write on.
        telescope_model: dict of TelescopeModel
            Dictionary of TelescopeModel's instances as used by the ArrayModel instance.
        site_model: Site model
            Site model.
        """
        with open(config_file_path, "w", encoding="utf-8") as file:
            self._write_header(file, "ARRAY CONFIGURATION FILE")

            # Be careful with the formatting - simtel is sensitive
            file.write("#ifndef TELESCOPE\n")
            file.write("# define TELESCOPE 0\n")
            file.write("#endif\n\n")

            # TELESCOPE 0 - global parameters
            file.write("#if TELESCOPE == 0\n")
            file.write(self.TAB + "echo *****************************\n")
            file.write(self.TAB + f"echo Site: {self._site}\n")
            file.write(self.TAB + f"echo LayoutName: {self._layout_name}\n")
            file.write(self.TAB + f"echo ModelVersion: {self._model_version}\n")
            file.write(self.TAB + "echo *****************************\n\n")

            # Writing site parameters
            self._write_site_parameters(file, site_model)

            # Maximum telescopes
            file.write(self.TAB + f"maximum_telescopes = {len(telescope_model)}\n\n")

            # Default telescope in sim_telarray - 0th tel in telescope list
            _, first_telescope = next(iter(telescope_model.items()))
            tel_config_file = first_telescope.get_config_file(no_export=True).name
            file.write(f"# include <{tel_config_file}>\n\n")

            # Looping over telescopes
            for count, (tel_name, tel_model) in enumerate(telescope_model.items()):
                tel_config_file = tel_model.get_config_file(no_export=True).name
                file.write(f"%{tel_name}\n")
                file.write(f"#elif TELESCOPE == {count + 1}\n\n")
                file.write(f"# include <{tel_config_file}>\n\n")
            file.write("#endif \n\n")

    def write_single_mirror_list_file(
        self, mirror_number, mirrors, single_mirror_list_file, set_focal_length_to_zero=False
    ):
        """
        Write the sim_telarray mirror list file for a single mirror.

        Parameters
        ----------
        mirror_number: int
            Mirror number.
        mirrors: Mirrors
            Instance of Mirrors.
        single_mirror_list_file: str or Path
            Path of the file to write on.
        set_focal_length_to_zero: bool
            Flag to set the focal length to zero.
        """
        (
            __,
            __,
            mirror_panel_diameter,
            focal_length,
            shape_type,
        ) = mirrors.get_single_mirror_parameters(mirror_number)
        with open(single_mirror_list_file, "w", encoding="utf-8") as file:
            self._write_header(file, "MIRROR LIST FILE", "#")

            file.write("# Column 1: X pos. [cm] (North/Down)\n")
            file.write("# Column 2: Y pos. [cm] (West/Right from camera)\n")
            file.write("# Column 3: flat-to-flat diameter [cm]\n")
            file.write(
                "# Column 4: focal length [cm], typically zero = adapting in sim_telarray.\n"
            )
            file.write(
                "# Column 5: shape type: 0=circular, 1=hex. with flat side parallel to y, "
                "2=square, 3=other hex. (default: 0)\n"
            )
            file.write(
                "# Column 6: Z pos (height above dish backplane) [cm], typ. omitted (or zero)"
                " to adapt to dish shape settings.\n"
            )
            file.write("#\n")
            file.write(
                f"0. 0. {mirror_panel_diameter.to('cm').value} "
                f"{focal_length.to('cm').value if not set_focal_length_to_zero else 0} "
                f"{shape_type} 0.\n"
            )

    def _write_header(self, file, title, comment_char="%"):
        """
        Write a generic header.

        Parameters
        ----------
        file: file
            File to write.
        title: str
            Title of the header.
        comment_char: str
            Character to be used for comments, which differs among ctypes of config files.
        """
        header = f"{comment_char}{50 * '='}\n"
        header += f"{comment_char} {title}\n"
        header += f"{comment_char} Site: {self._site}\n"
        header += f"{comment_char} ModelVersion: {self._model_version}\n"
        header += (
            f"{comment_char} TelescopeModelName: {self._telescope_model_name}\n"
            if self._telescope_model_name is not None
            else ""
        )
        header += (
            f"{comment_char} LayoutName: {self._layout_name}\n"
            if self._layout_name is not None
            else ""
        )
        header += f"{comment_char} Label: {self._label}\n" if self._label is not None else ""
        header += f"{comment_char}{50 * '='}\n"
        header += f"{comment_char}\n"
        file.write(header)

    def _write_site_parameters(self, file, site_model):
        """
        Write site parameters.

        Parameters
        ----------
        file: file
            File to write on.
        site_model: SiteModel
            Site model.
        """
        file.write(self.TAB + "% Site parameters\n")
        _site_parameters = site_model.get_simtel_parameters()
        for par, value in _site_parameters.items():
            _simtel_name = names.get_simulation_software_name_from_parameter_name(
                par,
                simulation_software="sim_telarray",
                search_telescope_parameters=False,
                search_site_parameters=True,
            )
            if _simtel_name is not None:
                file.write(f"{self.TAB}{_simtel_name} = {value}\n")
        _simtel_meta = self._get_simtel_metadata("site")
        for _simtel_name, value in _simtel_meta.items():
            file.write(f"{self.TAB}{_simtel_name} = {value}\n")
        file.write("\n")
