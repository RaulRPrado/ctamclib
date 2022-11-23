import logging
import shutil
from copy import copy
from pydoc import locate

import astropy.io.ascii
import numpy as np
from astropy import units as u
from astropy.table import Table

import simtools.util.general as gen
from simtools import db_handler, io_handler
from simtools.model.camera import Camera
from simtools.model.mirrors import Mirrors
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.util import names

__all__ = ["InvalidParameter", "TelescopeModel"]


class InvalidParameter(Exception):
    pass


class TelescopeModel:
    """
    TelescopeModel represents the MC model of an individual telescope. It contains the list of \
    parameters that can be read from the DB. A set of methods are available to manipulate \
    parameters (changing, adding, removing etc).

    Parameters
    ----------
    site: (str, required)
        South or North.
    telescope_model_name: (str, required)
        Telescope name (ex. LST-1, ...).
    mongo_db_config: (dict, required)
        MongoDB configuration.
    model_version: (str, optional)
        Version of the model (ex. prod5) (default is 'Current').
    label: (str, optional)
        Instance label. Important for output file naming (default is None).
    """

    def __init__(
        self,
        site,
        telescope_model_name,
        mongo_db_config,
        model_version="Current",
        label=None,
    ):
        """
        Initialize TelescopeModel.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init TelescopeModel")

        self.site = names.validate_site_name(site)
        self.name = names.validate_telescope_model_name(telescope_model_name)
        self.model_version = names.validate_model_version_name(model_version)
        self.label = label
        self._extra_label = None

        self.io_handler = io_handler.IOHandler()
        self.mongo_db_config = mongo_db_config

        self._parameters = dict()

        self._load_parameters_from_db()

        self._set_config_file_directory_and_name()
        self._is_config_file_up_to_date = False
        self._is_exported_model_files_up_to_date = False

    @property
    def mirrors(self):
        """
        Load the mirror information if the class instance hasn't done it yet.
        """
        if not hasattr(self, "_mirrors"):
            self._load_mirrors()
        return self._mirrors

    @property
    def camera(self):
        """
        Load the camera information if the class instance hasn't done it yet.
        """
        if not hasattr(self, "_camera"):
            self._load_camera()
        return self._camera

    @property
    def reference_data(self):
        """
        Load the reference data information if the class instance hasn't done it yet.
        """
        if not hasattr(self, "_reference_data"):
            self._load_reference_data()
        return self._reference_data

    @property
    def derived(self):
        """
        Load the derived values and export them if the class instance hasn't done it yet.
        """
        if not hasattr(self, "_derived"):
            self._load_derived_values()
            self.export_derived_files()
        return self._derived

    @property
    def extra_label(self):
        """
        Return the extra label if defined, if not return ''.
        """
        return self._extra_label if self._extra_label is not None else ""

    @classmethod
    def from_config_file(cls, config_file_name, site, telescope_model_name, label=None):
        """
        Create a TelescopeModel from a sim_telarray config file.

        Notes
        -----
        Todo: Dealing with ifdef/indef etc. By now it just keeps the last version of the parameters
        in the file.

        Parameters
        ----------
        config_file_name: (str or Path, required)
            Path to the input config file.
        site: (str. required)
            South or North.
        telescope_model_name: (str, required)
            Telescope model name for the base set of parameters (ex. LST-1, ...).
        label: (str, optional)
            Instance label. Important for output file naming (default is None).

        Returns
        -------
        TelescopeModel
            Instance of TelescopeModel.
        """
        parameters = dict()
        tel = cls(
            site=site,
            telescope_model_name=telescope_model_name,
            mongo_db_config=None,
            label=label,
        )

        def _process_line(words):
            """
            Process a line of the input config file that contains a parameter.

            Parameters
            ----------
            words: list of str
                List of str from the split of a line from the file.

            Returns
            -------
            (par_name, par_value)
            """
            i_comment = len(words)  # Index of any possible comment
            for w in words:
                if "%" in w:
                    i_comment = words.index(w)
                    break
            words = words[0:i_comment]  # Removing comment
            par_name = words[0].replace("=", "")
            par_value = ""
            for w in words[1:]:
                w = w.replace("=", "")
                w = w.replace(",", " ")
                par_value += w + " "
            par_value = par_value.rstrip().lstrip()  # Removing trailing spaces (left and right)
            return par_name, par_value

        with open(config_file_name, "r") as file:
            for line in file:
                words = line.split()
                if len(words) == 0:
                    continue
                elif "%" in words[0] or "echo" in words:
                    continue
                elif "#" not in line and len(words) > 0:
                    par, value = _process_line(words)
                    parameters[par] = value

        for par, value in parameters.items():
            tel.add_parameter(par, value)

        tel._is_exported_model_files_up_to_date = True
        return tel

    def set_extra_label(self, extra_label):
        """
        Set an extra label for the name of the config file.

        Notes
        -----
        The config file directory name is not affected by the extra label. Only the file name is \
        changed. This is important for the ArrayModel class to export multiple config files in the\
        same directory.

        Parameters
        ----------
        extra_label: (str, required)
            Extra label to be appended to the original label.
        """

        self._extra_label = extra_label
        self._set_config_file_directory_and_name()

    def _set_config_file_directory_and_name(self):
        """Define the variable _config_file_directory and create directories, if needed."""

        self._config_file_directory = self.io_handler.get_output_directory(self.label, "model")

        # Setting file name and the location
        config_file_name = names.simtel_telescope_config_file_name(
            self.site, self.name, self.model_version, self.label, self._extra_label
        )
        self._config_file_path = self._config_file_directory.joinpath(config_file_name)

    def _load_parameters_from_db(self):
        """Read parameters from DB and store them in _parameters."""

        if self.mongo_db_config is None:
            return

        self._logger.debug("Reading telescope parameters from DB")

        self._set_config_file_directory_and_name()
        db = db_handler.DatabaseHandler(mongo_db_config=self.mongo_db_config)
        self._parameters = db.get_model_parameters(
            self.site, self.name, self.model_version, only_applicable=True
        )

        self._logger.debug("Reading site parameters from DB")
        _site_pars = db.get_site_parameters(self.site, self.model_version, only_applicable=True)
        self._parameters.update(_site_pars)

    def has_parameter(self, par_name):
        """
        Verify if the parameter is in the model.

        Parameters
        ----------
        par_name: (str, required)
            Name of the parameter.

        Returns
        -------
        bool
            True if parameter is in the model.
        """
        return par_name in self._parameters

    def get_parameter(self, par_name):
        """
        Get an existing parameter of the model, including derived parameters.

        Parameters
        ----------
        par_name: (str, required)
            Name of the parameter.

        Returns
        -------
        Value of the parameter

        Raises
        ------
        InvalidParameter
            If par_name does not match any parameter in this model.
        """
        try:
            return self._parameters[par_name]
        except KeyError:
            pass  # search in the derived parameters
        try:
            return self.derived[par_name]
        except KeyError:
            msg = f"Parameter {par_name} was not found in the model"
            self._logger.error(msg)
            raise InvalidParameter(msg)

    def get_parameter_value(self, par_name):
        """
        Get the value of an existing parameter of the model.

        Parameters
        ----------
        par_name: (str, required)
            Name of the parameter.

        Returns
        -------
        Value of the parameter.

        Raises
        ------
        InvalidParameter
            If par_name does not match any parameter in this model.
        """
        par_info = self.get_parameter(par_name)
        return par_info["Value"]

    def get_parameter_value_with_unit(self, par_name):
        """
        Get the value of an existing parameter of the model as an Astropy Quantity with its unit.\
        If no unit is provided in the model, the value is returned without a unit.

        Parameters
        ----------
        par_name: (str, required)
            Name of the parameter.

        Returns
        -------
        Astropy quantity with the value of the parameter multiplied by its unit. If no unit is \
        provided in the model, the value is returned without a unit.

        Raises
        ------
        InvalidParameter
            If par_name does not match any parameter in this model.
        """
        par_info = self.get_parameter(par_name)
        if "units" in par_info:
            return par_info["Value"] * u.Unit(par_info["units"])
        return par_info["Value"]

    def add_parameter(self, par_name, value, is_file=False, is_aplicable=True):
        """
        Add a new parameters to the model. This function does not modify the DB, it affects only \
        the current instance.

        Parameters
        ----------
        par_name: (str, required)
            Name of the parameter.
        value: (required)
            Value of the parameter.
        is_file: (bool, optional)
            Indicates whether the new parameter is a file or not (default is False).
        is_aplicable: (bool, optional)
            Indicates whether the new parameter is applicable or not (default is True).

        Raises
        ------
        InvalidParameter
            If an existing parameter is tried to be added.
        """
        if par_name in self._parameters:
            msg = "Parameter {} already in the model, use change_parameter instead".format(par_name)
            self._logger.error(msg)
            raise InvalidParameter(msg)
        else:
            self._logger.info("Adding {}={} to the model".format(par_name, value))
            self._parameters[par_name] = dict()
            self._parameters[par_name]["Value"] = value
            self._parameters[par_name]["Type"] = type(value)
            self._parameters[par_name]["Applicable"] = is_aplicable
            self._parameters[par_name]["File"] = is_file

        self._is_config_file_up_to_date = False
        if is_file:
            self._is_exported_model_files_up_to_date = False

    def change_parameter(self, par_name, value):
        """
        Change the value of an existing parameter to the model. This function does not modify the \
        DB, it affects only the current instance.

        Parameters
        ----------
        par_name: (str, required)
            Name of the parameter.
        value: (required)
            Value of the parameter.

        Raises
        ------
        InvalidParameter
            If the parameter to be changed does not exist in this model.
        """
        if par_name not in self._parameters:
            msg = "Parameter {} not in the model, use add_parameters instead".format(par_name)
            self._logger.error(msg)
            raise InvalidParameter(msg)
        else:
            type_of_par_name = locate(self._parameters[par_name]["Type"])
            if not isinstance(value, type_of_par_name):
                self._logger.warning(
                    f"The type of the provided value ({value}, {type(value)}) "
                    f"is different from the type of {par_name} "
                    f"({self._parameters[par_name]['Type']}). "
                    f"Attempting to cast to the correct type."
                )
                try:
                    value = type_of_par_name(value)
                except ValueError:
                    self._logger.error(
                        f"Could not cast {value} to {self._parameters[par_name]['Type']}."
                    )
                    raise

            self._logger.debug(
                f"Changing parameter {par_name} "
                f"from {self._parameters[par_name]['Value']} to {value}"
            )
            self._parameters[par_name]["Value"] = value

            # In case parameter is a file, the model files will be outdated
            if self._parameters[par_name]["File"]:
                self._is_exported_model_files_up_to_date = False

        self._is_config_file_up_to_date = False

    def change_multiple_parameters(self, **kwargs):
        """
        Change the value of multiple existing parameters in the model. This function does not \
        modify the DB, it affects only the current instance.

        Parameters
        ----------
        **kwargs
            Parameters should be passed as parameter_name=value.

        Raises
        ------
        InvalidParameter
            If at least one of the parameters to be changed does not exist in this model.
        """
        for par, value in kwargs.items():
            if par in self._parameters:
                self.change_parameter(par, value)
            else:
                self.add_parameter(par, value)

        self._is_config_file_up_to_date = False

    def remove_parameters(self, *args):
        """
        Remove a set of parameters from the model.

        Parameters
        ----------
        *args
            Each parameter to be removed has to be passed as args.

        Raises
        ------
        InvalidParameter
            If at least one of the parameter to be removed is not in this model.
        """
        for par in args:
            if par in self._parameters:
                self._logger.info("Removing parameter {}".format(par))
                del self._parameters[par]
            else:
                msg = "Could not remove parameter {} because it does not exist".format(par)
                self._logger.error(msg)
                raise InvalidParameter(msg)
        self._is_config_file_up_to_date = False

    def add_parameter_file(self, par_name, file_path):
        """
        Add a file to the config file directory.

        Parameters
        ----------
        par_name: (str, required)
            Name of the parameter.
        file_path: (str, required)
            Path of the file to be added to the config file directory.
        """
        if not hasattr(self, "_added_parameter_files"):
            self._added_parameter_files = list()
        self._added_parameter_files.append(par_name)
        shutil.copy(file_path, self._config_file_directory)

    def export_model_files(self):
        """Exports the model files into the config file directory."""
        db = db_handler.DatabaseHandler(mongo_db_config=self.mongo_db_config)

        # Removing parameter files added manually (which are not in DB)
        pars_from_db = copy(self._parameters)
        if hasattr(self, "_added_parameter_files"):
            for par in self._added_parameter_files:
                pars_from_db.pop(par)

        db.export_model_files(pars_from_db, self._config_file_directory)
        self._is_exported_model_files_up_to_date = True

    def print_parameters(self):
        """Print parameters and their values for debugging purposes."""
        for par, info in self._parameters.items():
            print("{} = {}".format(par, info["Value"]))

    def export_config_file(self):
        """Export the config file used by sim_telarray."""

        # Exporting model file
        if not self._is_exported_model_files_up_to_date:
            self.export_model_files()

        # Using SimtelConfigWriter to write the config file.
        self._load_simtel_config_writer()
        self.simtel_config_writer.write_telescope_config_file(
            config_file_path=self._config_file_path, parameters=self._parameters
        )

    def export_derived_files(self):
        """Write to disk a file from the derived values DB."""

        db = db_handler.DatabaseHandler(mongo_db_config=self.mongo_db_config)
        for par_now in self.derived:
            if self.derived[par_now]["File"]:
                db.export_file_db(
                    db_name=db.DB_DERIVED_VALUES,
                    dest=self.io_handler.get_output_directory(self.label, "derived"),
                    file_name=self.derived[par_now]["Value"],
                )

    def get_config_file(self, no_export=False):
        """
        Get the path of the config file for sim_telarray. The config file is produced if the file\
        is not updated.

        Parameters
        ----------
        no_export: (bool, optional)
            Turn it on if you do not want the file to be exported (default is False).

        Returns
        -------
        Path
            Path of the exported config file for sim_telarray.
        """
        if not self._is_config_file_up_to_date and not no_export:
            self.export_config_file()
        return self._config_file_path

    def get_config_directory(self):
        """
        Get the path where all the configuration files for sim_telarray are written to.

        Returns
        -------
        Path
            Path where all the configuration files for sim_telarray are written to.
        """
        return self._config_file_directory

    def get_derived_directory(self):
        """
        Get the path where all the files with derived values for are written to.

        Returns
        -------
        Path
            Path where all the files with derived values are written to.
        """
        return self._config_file_directory.parents[0].joinpath("derived")

    def get_telescope_transmission_parameters(self):
        """
        Get tel. transmission pars as a list of floats.

        Returns
        -------
        list of floats
            List of 4 parameters that describe the tel. transmission vs off-axis.
        """

        telescope_transmission = self.get_parameter_value("telescope_transmission")
        if isinstance(telescope_transmission, str):
            return [float(v) for v in self.get_parameter_value("telescope_transmission").split()]
        else:
            return [float(telescope_transmission), 0, 0, 0]

    def export_single_mirror_list_file(self, mirror_number, set_focal_length_to_zero):
        """
        Export a mirror list file with a single mirror in it.

        Parameters
        ----------
        mirror_number: (int, required)
            Number index of the mirror.
        set_focal_length_to_zero: (bool, required)
            Set the focal length to zero if True.
        """
        if mirror_number > self.mirrors.number_of_mirrors:
            logging.error("mirror_number > number_of_mirrors")
            return None

        file_name = names.simtel_single_mirror_list_file_name(
            self.site, self.name, self.model_version, mirror_number, self.label
        )
        if not hasattr(self, "_single_mirror_list_file_path"):
            self._single_mirror_list_file_paths = dict()
        self._single_mirror_list_file_paths[mirror_number] = self._config_file_directory.joinpath(
            file_name
        )

        # Using SimtelConfigWriter
        self._load_simtel_config_writer()
        self.simtel_config_writer.write_single_mirror_list_file(
            mirror_number,
            self.mirrors,
            self._single_mirror_list_file_paths[mirror_number],
            set_focal_length_to_zero,
        )

    # END of export_single_mirror_list_file

    def get_single_mirror_list_file(self, mirror_number, set_focal_length_to_zero=False):
        """
        Get the path to the single mirror list file.

        Parameters
        ----------
        mirror_number: (int, required)
            Mirror number.
        set_focal_length_to_zero: (bool, optional)
            Flag to set the focal length to zero (default is False).

        Returns
        -------
        Path
            Path of the single mirror list file.
        """
        self.export_single_mirror_list_file(mirror_number, set_focal_length_to_zero)
        return self._single_mirror_list_file_paths[mirror_number]

    def _load_mirrors(self):
        """Load the attribute mirrors by creating a Mirrors object with the mirror list file."""
        mirror_list_file_name = self._parameters["mirror_list"]["Value"]
        try:
            mirror_list_file = gen.find_file(mirror_list_file_name, self._config_file_directory)
        except FileNotFoundError:
            mirror_list_file = gen.find_file(mirror_list_file_name, self.io_handler.model_path)
            self._logger.warning(
                "Mirror_list_file was not found in the config directory - "
                "Using the one found in the model_path"
            )
        self._mirrors = Mirrors(mirror_list_file)

    def _load_reference_data(self):
        """Load the reference data for this telescope from the DB."""
        self._logger.debug("Reading reference data from DB")
        db = db_handler.DatabaseHandler(mongo_db_config=self.mongo_db_config)
        self._reference_data = db.get_reference_data(
            self.site, self.model_version, only_applicable=True
        )

    def _load_derived_values(self):
        """Load the derived values for this telescope from the DB."""
        self._logger.debug("Reading derived data from DB")
        db = db_handler.DatabaseHandler(mongo_db_config=self.mongo_db_config)
        self._derived = db.get_derived_values(
            self.site,
            self.name,
            self.model_version,
        )

    def _load_camera(self):
        """Loading camera attribute by creating a Camera object with the camera config file."""
        camera_config_file = self.get_parameter_value("camera_config_file")
        focal_length = self.get_parameter_value("effective_focal_length")
        if focal_length == 0.0:
            self._logger.warning("Using focal_length because effective_focal_length is 0.")
            focal_length = self.get_parameter_value("focal_length")
        try:
            camera_config_file_path = gen.find_file(camera_config_file, self._config_file_directory)
        except FileNotFoundError:
            self._logger.warning(
                "The camera_config_file was not found in the config directory - "
                "Using the one found in the model_path"
            )
            camera_config_file_path = gen.find_file(camera_config_file, self.io_handler.model_path)

        self._camera = Camera(
            telescope_model_name=self.name,
            camera_config_file=camera_config_file_path,
            focal_length=focal_length,
        )

    def _load_simtel_config_writer(self):
        if not hasattr(self, "simtel_config_writer"):
            self.simtel_config_writer = SimtelConfigWriter(
                site=self.site,
                telescope_model_name=self.name,
                model_version=self.model_version,
                label=self.label,
            )

    def is_file_2D(self, par):
        """
        Check if the file referenced by par is a 2D table.

        Parameters
        ----------
        par: (str, required)
            Name of the parameter.

        Returns
        -------
        bool:
            True if the file is a 2D map type, False otherwise.
        """
        if not self.has_parameter(par):
            logging.error("Parameter {} does not exist".format(par))
            return False

        file_name = self.get_parameter_value(par)
        file = self.get_config_directory().joinpath(file_name)
        with open(file, "r") as f:
            is2D = "@RPOL@" in f.read()
        return is2D

    def read_two_dim_wavelength_angle(self, file_name):
        """
        Read a two dimensional distribution of wavelngth and angle (z-axis can be anything). Return\
        a dictionary with three arrays, wavelength, angles, z (can be transmission, reflectivity,\
        etc.)

        Parameters
        ----------
        file_name: (str or Path, required)
            File assumed to be in the model directory.

        Returns
        -------
        dict:
            dict of three arrays, wavelength, degrees, z.
        """

        _file = self.get_config_directory().joinpath(file_name)
        with open(_file, "r") as f:
            for i_line, line in enumerate(f):
                if line.startswith("ANGLE"):
                    degrees = np.array(line.strip().split("=")[1].split(), dtype=np.float16)
                    break  # The rest can be read with np.loadtxt

        _data = np.loadtxt(_file, skiprows=i_line + 1)

        return {
            "Wavelength": _data[:, 0],
            "Angle": degrees,
            "z": np.array(_data[:, 1:]).T,
        }

    def get_on_axis_eff_optical_area(self):
        """Return the on-axis effective optical area (derived previously for this telescope)."""

        ray_tracing_data = astropy.io.ascii.read(
            self.get_derived_directory().joinpath(self.derived["ray_tracing"]["Value"])
        )
        if not np.isclose(ray_tracing_data["Off-axis angle"][0], 0):
            self._logger.error(
                f"No value for the on-axis effective optical area exists."
                f" The minumum off-axis angle is {ray_tracing_data['Off-axis angle'][0]}"
            )
            raise ValueError
        return ray_tracing_data["eff_area"][0]

    def read_incidence_angle_distribution(self, incidence_angle_dist_file):
        """
        Read the incidence angle distribution from a file.

        Parameters
        ----------
        incidence_angle_dist_file: (str, required)
            File name of the incidence angle distribution

        Returns
        -------
        incidence_angle_dist: Astropy table
            Astropy table with the incidence angle distribution.
        """

        incidence_angle_dist = astropy.io.ascii.read(
            self.get_derived_directory().joinpath(incidence_angle_dist_file)
        )
        return incidence_angle_dist

    @staticmethod
    def calc_average_curve(curves, incidence_angle_dist):
        """
        Calculate an average curve from a set of curves, using as weights the distribution of \
        incidence angles provided in incidence_angle_dist.

        Parameters
        ----------
        curves: (dict, required)
            dict of with 3 "columns", Wavelength, Angle and z. The dictionary represents a two \
            dimensional distribution of wavelengths and angles with the z value being e.g.,\
             reflectivity, transmission, etc.
        incidence_angle_dist: (Astropy table, required)
            Astropy table with the incidence angle distribution. The assumed columns are "Incidence\
            angle" and "Fraction".

        Returns
        -------
        average_curve: astropy.table
            Instance of astropy Table with the averaged curve.
        """

        weights = list()
        for angle_now in curves["Angle"]:
            weights.append(
                incidence_angle_dist["Fraction"][
                    np.nanargmin(np.abs(angle_now - incidence_angle_dist["Incidence angle"].value))
                ]
            )

        average_curve = Table(
            [curves["Wavelength"], np.average(curves["z"], weights=weights, axis=0)],
            names=("Wavelength", "z"),
        )

        return average_curve

    def export_table_to_model_directory(self, file_name, table):
        """
        Write out a file with the provided table to the model directory.

        Parameters
        ----------
        file_name: (str, required)
            File name to write to.
        table: (astropy.table, required)
            Instance of astropy table with the values to write to the file.

        Returns
        -------
        Path:
            Path to the file exported.
        """

        file_to_write_to = self._config_file_directory.joinpath(file_name)
        table.write(file_to_write_to, format="ascii.commented_header", overwrite=True)
        return file_to_write_to.absolute()
