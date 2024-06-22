import copy
import logging
from pathlib import Path

import astropy.units as u
import numpy as np

import simtools.utils.general as gen
from simtools.io_operations import io_handler
from simtools.utils.general import collect_data_from_file_or_dict

__all__ = [
    "CorsikaConfig",
    "MissingRequiredInputInCorsikaConfigDataError",
    "InvalidCorsikaInputError",
]


class MissingRequiredInputInCorsikaConfigDataError(Exception):
    """Exception for missing required input in corsika config data."""


class InvalidCorsikaInputError(Exception):
    """Exception for invalid corsika input."""


class CorsikaConfig:
    """
    CorsikaConfig deals with configuration for running CORSIKA. User parameters must be given by \
    the corsika_config_data or corsika_config_file arguments. An example of corsika_config_data
    follows below.

    .. code-block:: python

        corsika_config_data = {
            'data_directory': .
            'primary': 'proton',
            'nshow': 10000,
            'nrun': 1,
            'zenith': 20 * u.deg,
            'viewcone': 5 * u.deg,
            'erange': [10 * u.GeV, 100 * u.TeV],
            'eslope': -2,
            'phi': 0 * u.deg,
            'cscat': [10, 1500 * u.m, 0]
        }

    Parameters
    ----------
    array_model: ArrayModel
        Array model.
    label: str
        Instance label.
    corsika_config_data: dict
        CORSIKA user parameters.
    corsika_config_file: str
        Name of the yaml configuration file. If not provided, \
        data/parameters/corsika_parameters.yml will be used.
    corsika_parameters_file: str
        Name of the yaml file to set remaining CORSIKA parameters.
    simtel_source_path: str or Path
        Location of source of the sim_telarray/CORSIKA package.
    """

    def __init__(
        self,
        array_model,
        label=None,
        corsika_config_data=None,
        corsika_config_file=None,
        corsika_parameters_file=None,
        simtel_source_path=None,
    ):
        """Initialize CorsikaConfig."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaConfig")

        self.label = label
        self.primary = None
        self.eslope = None
        self.config_file_path = None
        self._output_generic_file_name = None
        self._simtel_source_path = simtel_source_path

        self.io_handler = io_handler.IOHandler()

        self.array_model = array_model
        self._logger.debug(f"Building ArrayLayout {self.array_model.layout_name}")

        # Load parameters
        if corsika_parameters_file is None:
            corsika_parameters_file = self.io_handler.get_input_data_file(
                "parameters", "corsika_parameters.yml"
            )

        self._corsika_parameters = self.load_corsika_parameters_file(corsika_parameters_file)

        corsika_config_data = collect_data_from_file_or_dict(
            corsika_config_file, corsika_config_data
        )
        self.set_user_parameters(corsika_config_data)
        self._is_file_updated = False

    def __repr__(self):
        text = (
            f"<class {self.__class__.__name__}> "
            f"(site={self.array_model.site}, "
            f"layout={self.array_model.layout_name}, label={self.label})"
        )
        return text

    @staticmethod
    def load_corsika_parameters_file(corsika_parameters_file):
        """
        Load CORSIKA parameters from the provided corsika_parameters_file.

        Parameters
        ----------
        corsika_parameters_file: str or Path
            File with CORSIKA parameters.

        Returns
        -------
        corsika_parameters: dict
            Dictionary with CORSIKA parameters.
        """
        logger = logging.getLogger(__name__)
        logger.debug(f"Loading CORSIKA parameters from file {corsika_parameters_file}")
        corsika_parameters = gen.collect_data_from_file_or_dict(
            file_name=corsika_parameters_file, in_dict=None
        )
        return corsika_parameters

    def _collect_parameters(self, corsika_config_data, user_pars):
        for key_args, value_args in corsika_config_data.items():
            is_identified = False
            for par_name, par_info in user_pars.items():
                if key_args.upper() != par_name and key_args.upper() not in par_info["names"]:
                    continue
                validated_value_args = self._validate_and_convert_argument(
                    par_name, par_info, value_args
                )
                self._user_parameters[par_name] = validated_value_args
                is_identified = True
                break  # Break the inner loop if a match is found
            if not is_identified:
                self._raise_invalid_input_error(key_args)

    def _fill_default_parameters(self, user_pars):
        for par_name, par_info in user_pars.items():
            if par_name not in self._user_parameters:
                if "default" in par_info.keys():
                    validated_value = self._validate_and_convert_argument(
                        par_name, par_info, par_info["default"]
                    )
                    self._user_parameters[par_name] = validated_value
                else:
                    self._raise_missing_required_error(par_name)

    def _convert_azm_to_phip(self):
        phip = 180.0 - self._user_parameters["AZM"][0]
        phip = phip + 360.0 if phip < 0.0 else phip
        phip = phip - 360.0 if phip >= 360.0 else phip
        self._user_parameters["PHIP"] = [phip, phip]

    def _raise_invalid_input_error(self, key_args):
        msg = f"Argument {key_args} cannot be identified."
        self._logger.error(msg)
        raise InvalidCorsikaInputError(msg)

    def _raise_missing_required_error(self, par_name):
        msg = f"Required parameters {par_name} was not given (there may be more)."
        self._logger.error(msg)
        raise MissingRequiredInputInCorsikaConfigDataError(msg)

    def set_user_parameters(self, corsika_config_data):
        """
        Set user parameters from a dict.

        Parameters
        ----------
        corsika_config_data: dict
            Contains the user parameters. Ex.

            .. code-block:: python

                corsika_config_data = {
                    'primary': 'proton',
                    'nshow': 10000,
                    'nrun': 1,
                    'zenith': 20 * u.deg,
                    'viewcone': 5 * u.deg,
                    'erange': [10 * u.GeV, 100 * u.TeV],
                    'eslope': -2,
                    'phi': 0 * u.deg,
                    'cscat': [10, 1500 * u.m, 0]
                }

        Raises
        ------
        InvalidCorsikaInputError
            If any parameter given as input has wrong len, unit or
            an invalid name.
        MissingRequiredInputInCorsikaConfigDataError
            If any required user parameter is missing.
        """
        self._logger.debug("Setting user parameters from corsika_config_data")
        self._user_parameters = {}
        user_pars = self._corsika_parameters["USER_PARAMETERS"]

        self._collect_parameters(corsika_config_data, user_pars)
        self._fill_default_parameters(user_pars)

        if "AZM" in self._user_parameters:
            self._convert_azm_to_phip()

        self._is_file_updated = False

    def _fix_single_value_parameters(self, par_name, par_info, values):
        if len(values) == 1 and par_name in ["THETAP", "AZM"]:
            return values * 2
        if len(values) == 1 and par_name == "VIEWCONE":
            return [0.0 * u.Unit(par_info["unit"][0]), values[0]]
        return values

    def _handle_special_parameters(self, par_name, values):
        if par_name == "PRMPAR":
            return self._convert_primary_input_and_store_primary_name(values)
        if par_name == "ESLOPE":
            self.eslope = values[0]
        return values

    def _validate_length(self, par_name, par_info, values):
        if len(values) != par_info["len"]:
            msg = f"CORSIKA input entry with wrong len: {par_name}"
            self._logger.error(msg)
            raise InvalidCorsikaInputError(msg)

    def _convert_units(self, par_name, values, units):
        result = []
        for value, unit in zip(values, units):
            if unit is None:
                result.append(value)
                continue
            value = self._convert_to_quantity(value)
            if not value.unit.is_equivalent(unit):
                msg = f"CORSIKA input given with wrong unit: {par_name}"
                self._logger.error(msg)
                raise InvalidCorsikaInputError(msg)
            result.append(value.to(unit).value)
        return result

    def _convert_to_quantity(self, value):
        if isinstance(value, str):
            value = u.Quantity(value)
        if not isinstance(value, u.Quantity):
            msg = "CORSIKA input given without unit"
            self._logger.error(msg)
            raise InvalidCorsikaInputError(msg)
        return value

    def _validate_and_convert_argument(self, par_name, par_info, value_args_in):
        """
        Validate input user parameter and convert it to the right units, if needed.

        Returns the validated arguments in a list.

        Parameters
        ----------
        par_name: str
            Name of the parameter as used in the CORSIKA input file (e.g. PRMPAR, THETAP ...).

        par_info: dict
            Dictionary with parameter data.

        value_args_in: list
            List of values for the parameter.
        """
        value_args = self._convert_to_quantities(value_args_in)
        value_args = self._fix_single_value_parameters(par_name, par_info, value_args)
        value_args = self._handle_special_parameters(par_name, value_args)
        self._validate_length(par_name, par_info, value_args)

        if "unit" not in par_info:
            return value_args

        par_unit = gen.copy_as_list(par_info["unit"])
        return self._convert_units(par_name, value_args, par_unit)

    def _convert_primary_input_and_store_primary_name(self, value):
        """
        Convert a primary name into the proper CORSIKA particle ID and store its name in \
        the self.primary attribute.

        Parameters
        ----------
        value: str
            Input primary name (e.g gamma, proton ...)

        Raises
        ------
        InvalidPrimary
            If the input name is not found.

        Returns
        -------
        int
            Respective number of the given primary.
        """
        for prim_name, prim_info in self._corsika_parameters["PRIMARIES"].items():
            if value[0].upper() == prim_name or value[0].upper() in prim_info["names"]:
                self.primary = prim_name.lower()
                return [prim_info["number"]]
        msg = f"Primary not valid: {value}"
        self._logger.error(msg)
        raise InvalidCorsikaInputError(msg)

    def get_user_parameter(self, par_name):
        """
        Get the value of a user parameter.

        Parameters
        ----------
        par_name: str
            Name of the parameter as used in the CORSIKA input file (e.g. PRMPAR, THETAP ...)

        Raises
        ------
        KeyError
            When par_name is not a valid parameter name.

        Returns
        -------
        list
            Value(s) of the parameter.
        """
        try:
            par_value = self._user_parameters[par_name.upper()]
        except KeyError:
            self._logger.warning(f"Parameter {par_name} is not a user parameter")
            raise

        return par_value if len(par_value) > 1 else par_value[0]

    def print_user_parameters(self):
        """Print user parameters for inspection."""
        for par, value in self._user_parameters.items():
            print(f"{par} = {value}")

    def export_input_file(self, use_multipipe=False):
        """
        Create and export CORSIKA input file.

        Parameters
        ----------
        use_multipipe: bool
            Whether to set the CORSIKA Inputs file to pipe
            the output directly to sim_telarray or not.
        """
        sub_dir = "corsika_simtel" if use_multipipe else "corsika"
        self._set_output_file_and_directory(sub_dir)
        self._logger.debug(f"Exporting CORSIKA input file to {self.config_file_path}")

        def _get_text_single_line(pars):
            text = ""
            for par, values in pars.items():
                line = par + " "
                for v in values:
                    line += str(v) + " "
                line += "\n"
                text += line
            return text

        def _get_text_multiple_lines(pars):
            text = ""
            for par, value_list in pars.items():
                for value in value_list:
                    new_pars = {par: value}
                    text += _get_text_single_line(new_pars)
            return text

        with open(self.config_file_path, "w", encoding="utf-8") as file:
            file.write("\n* [ RUN PARAMETERS ]\n")
            # Removing AZM entry first
            _user_pars_temp = copy.copy(self._user_parameters)
            _user_pars_temp.pop("AZM")
            text_parameters = _get_text_single_line(_user_pars_temp)
            file.write(text_parameters)

            file.write("\n* [ SITE PARAMETERS ]\n")
            text_site_parameters = _get_text_single_line(
                self.array_model.site_model.get_corsika_site_parameters(config_file_style=True)
            )
            file.write(text_site_parameters)

            # Defining the IACT variables for the output file name
            file.write("\n")
            file.write(f"IACT setenv PRMNAME {self.primary}\n")
            file.write(f"IACT setenv ZA {int(self._user_parameters['THETAP'][0])}\n")
            file.write(f"IACT setenv AZM {int(self._user_parameters['AZM'][0])}\n")

            file.write("\n* [ SEEDS ]\n")
            self._write_seeds(file)

            file.write("\n* [ TELESCOPES ]\n")
            telescope_list_text = self.get_corsika_input_list()
            file.write(telescope_list_text)

            file.write("\n* [ INTERACTION FLAGS ]\n")
            text_interaction_flags = _get_text_single_line(
                self._corsika_parameters["INTERACTION_FLAGS"]
            )
            file.write(text_interaction_flags)

            file.write("\n* [ CHERENKOV EMISSION PARAMETERS ]\n")
            text_cherenkov = _get_text_single_line(
                self._corsika_parameters["CHERENKOV_EMISSION_PARAMETERS"]
            )
            file.write(text_cherenkov)

            file.write("\n* [ DEBUGGING OUTPUT PARAMETERS ]\n")
            text_debugging = _get_text_single_line(
                self._corsika_parameters["DEBUGGING_OUTPUT_PARAMETERS"]
            )
            file.write(text_debugging)

            file.write("\n* [ OUTPUT FILE ]\n")
            if use_multipipe:
                run_cta_script = Path(self.config_file_path.parent).joinpath("run_cta_multipipe")
                file.write(f"TELFIL |{str(run_cta_script)}\n")
            else:
                file.write(f"TELFIL {self._output_generic_file_name}\n")

            file.write("\n* [ IACT TUNING PARAMETERS ]\n")
            text_iact = _get_text_multiple_lines(self._corsika_parameters["IACT_TUNING_PARAMETERS"])
            file.write(text_iact)

            file.write("\nEXIT")

        self._is_file_updated = True

    def get_file_name(self, file_type, run_number=None):
        """
        Get a CORSIKA config style file name for various file types.

        Parameters
        ----------
        file_type: str
            The type of file (determines the file suffix).
            Choices are config_tmp, config or output_generic.
        run_number: int
            Run number.

        Returns
        -------
        str
            for file_type="config_tmp":
                Get the CORSIKA input file for one specific run.
                This is the input file after being pre-processed by sim_telarray (pfp).
            for file_type="config":
                Get a general CORSIKA config inputs file.
            for file_type="output_generic"
                Get a generic file name for the TELFIL option in the CORSIKA inputs file.
            for file_type="multipipe"
                Get a multipipe "file name" for the TELFIL option in the CORSIKA inputs file.

        Raises
        ------
        ValueError
            If file_type is unknown or if the run number is not given for file_type==config_tmp.
        """
        file_label = f"_{self.label}" if self.label is not None else ""
        view_cone = ""
        if self._user_parameters["VIEWCONE"][0] != 0 or self._user_parameters["VIEWCONE"][1] != 0:
            view_cone = (
                f"_cone{int(self._user_parameters['VIEWCONE'][0]):d}-"
                f"{int(self._user_parameters['VIEWCONE'][1]):d}"
            )
        file_name = (
            f"{self.primary}_{self.array_model.site}_{self.array_model.layout_name}_"
            f"za{int(self._user_parameters['THETAP'][0]):03}-"
            f"azm{int(self._user_parameters['AZM'][0]):03}deg"
            f"{view_cone}{file_label}"
        )

        if file_type == "config_tmp":
            if run_number is not None:
                return f"corsika_config_run{run_number:06}_{file_name}.txt"
            raise ValueError("Must provide a run number for a temporary CORSIKA config file")
        if file_type == "config":
            return f"corsika_config_{file_name}.input"
        if file_type == "output_generic":
            # The XXXXXX will be replaced by the run number after the pfp step with sed
            file_name = (
                f"corsika_runXXXXXX_"
                f"{self.primary}_za{int(self._user_parameters['THETAP'][0]):03}deg_"
                f"azm{int(self._user_parameters['AZM'][0]):03}deg"
                f"_{self.array_model.site}_{self.array_model.layout_name}{file_label}.zst"
            )
            return file_name
        if file_type == "multipipe":
            return f"multi_cta-{self.array_model.site}-{self.array_model.layout_name}.cfg"

        raise ValueError(f"The requested file type ({file_type}) is unknown")

    def _set_output_file_and_directory(self, sub_dir="corsika"):
        config_file_name = self.get_file_name(file_type="config")
        file_directory = self.io_handler.get_output_directory(label=self.label, sub_dir=sub_dir)
        self._logger.info(f"Creating directory {file_directory}, if needed.")
        file_directory.mkdir(parents=True, exist_ok=True)
        self.config_file_path = file_directory.joinpath(config_file_name)

        self._output_generic_file_name = self.get_file_name(file_type="output_generic")

    def _write_seeds(self, file):
        """
        Generate and write seeds in the CORSIKA input file.

        Parameters
        ----------
        file: stream
            File where the telescope positions will be written.
        """
        random_seed = self._user_parameters["PRMPAR"][0] + self._user_parameters["RUNNR"][0]
        rng = np.random.default_rng(random_seed)
        corsika_seeds = [int(rng.uniform(0, 1e7)) for _ in range(4)]

        for s in corsika_seeds:
            file.write(f"SEED {s} 0 0\n")

    def get_input_file(self, use_multipipe=False):
        """
        Get the full path of the CORSIKA input file.

        Returns
        -------
        Path:
            Full path of the CORSIKA input file.
        """
        if not self._is_file_updated:
            self.export_input_file(use_multipipe)
        return self.config_file_path

    def _convert_to_quantities(self, value_args):
        """
        Convert a list of value, unit pairs into a list of astropy quantities.
        (note similarity to simtools.general.validate_config_data; unfortunately
        minor differences are required as CORSIKA is very specific about the
        input parameter representation).

        Parameters
        ----------
        value_args: list
            List of value/unit pairs (e.g., ["10 m", "20 m"])

        Returns
        -------
        list
            List of astropy quantities (or strings)
        """
        if isinstance(value_args, str):
            return [value_args]
        if isinstance(value_args, dict) and "value" in value_args and "unit" in value_args:
            return [value_args["value"] * u.Unit(value_args["unit"])]
        if isinstance(value_args, list):
            return [
                (
                    value
                    if isinstance(value, u.Quantity)
                    else (
                        value["value"] * u.Unit(value["unit"])
                        if isinstance(value, dict) and "value" in value and "unit" in value
                        else value
                    )
                )
                for value in value_args
            ]
        return [value_args]

    def get_corsika_input_list(self):
        """
        List of telescope positions in the format required for the CORSIKA input file.

        Returns
        -------
        str
            Piece of text to be added to the CORSIKA input file.
        """
        corsika_input_list = ""
        for telescope_name, telescope in self.array_model.telescope_model.items():
            positions = telescope.get_parameter_value_with_unit("array_element_position_ground")
            corsika_input_list += "TELESCOPE"
            for pos in positions:
                corsika_input_list += f"\t {pos.to('cm').value:.3f}"
            sphere_radius = telescope.get_parameter_value_with_unit("telescope_sphere_radius").to(
                "cm"
            )
            corsika_input_list += f"\t {sphere_radius:.3f}"
            corsika_input_list += f"\t # {telescope_name}\n"

        return corsika_input_list
