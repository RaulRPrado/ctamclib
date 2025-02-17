"""CORSIKA configuration."""

import logging
from pathlib import Path

import numpy as np
from astropy import units as u

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.io_operations import io_handler
from simtools.model.model_parameter import ModelParameter

__all__ = [
    "CorsikaConfig",
    "InvalidCorsikaInputError",
]


class InvalidCorsikaInputError(Exception):
    """Exception for invalid corsika input."""


class CorsikaConfig:
    """
    Configuration for the CORSIKA air shower simulation software.

    Follows closely the CORSIKA definitions and output format (see CORSIKA manual).

    The configuration is set as a dict corresponding to the command line configuration groups
    (especially simulation_software, simulation configuration, simulation parameters).

    Parameters
    ----------
    array_model : ArrayModel
        Array model.
    args_dict : dict
        Configuration dictionary.
    db_config : dict
        MongoDB configuration.
    label : str
        Instance label.
    """

    def __init__(self, array_model, args_dict, db_config=None, label=None):
        """Initialize CorsikaConfig."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaConfig")

        self.label = label
        self.zenith_angle = None
        self.azimuth_angle = None
        self._run_number = None
        self.config_file_path = None
        # The following uses the setter defined below, that is why the args_dict is passed
        self.primary_particle = args_dict

        self.io_handler = io_handler.IOHandler()
        self.array_model = array_model
        self.config = self.fill_corsika_configuration(args_dict, db_config)
        self._is_file_updated = False

    def __repr__(self):
        """CorsikaConfig class representation."""
        return (
            f"<class {self.__class__.__name__}> "
            f"(site={self.array_model.site}, "
            f"layout={self.array_model.layout_name}, label={self.label})"
        )

    @property
    def primary_particle(self):
        """Primary particle."""
        return self._primary_particle

    @primary_particle.setter
    def primary_particle(self, args_dict):
        """
        Set primary particle from input dictionary.

        This is to make sure that when setting the primary particle,
        we get the full PrimaryParticle object expected.

        Parameters
        ----------
        args_dict: dict
            Configuration dictionary
        """
        self._primary_particle = self._set_primary_particle(args_dict)

    def fill_corsika_configuration(self, args_dict, db_config=None):
        """
        Fill CORSIKA configuration.

        Dictionary keys are CORSIKA parameter names.
        Values are converted to CORSIKA-consistent units.


        Parameters
        ----------
        args_dict : dict
            Configuration dictionary.
        db_config: dict
            Database configuration.

        Returns
        -------
        dict
            Dictionary with CORSIKA parameters.
        """
        if args_dict is None:
            return {}

        self._is_file_updated = False
        self.azimuth_angle = int(args_dict["azimuth_angle"].to("deg").value)
        self.zenith_angle = args_dict["zenith_angle"].to("deg").value

        self._logger.debug(
            f"Setting CORSIKA parameters from database ({args_dict['model_version']})"
        )

        config = {}
        config["USER_INPUT"] = self._corsika_configuration_from_user_input(args_dict)

        if db_config is None:  # all following parameter require DB
            return config

        db_model_parameters = ModelParameter(
            mongo_db_config=db_config, model_version=args_dict["model_version"]
        )
        parameters_from_db = db_model_parameters.get_simulation_software_parameters("corsika")

        config["INTERACTION_FLAGS"] = self._corsika_configuration_interaction_flags(
            parameters_from_db
        )
        config["CHERENKOV_EMISSION_PARAMETERS"] = self._corsika_configuration_cherenkov_parameters(
            parameters_from_db
        )
        config["DEBUGGING_OUTPUT_PARAMETERS"] = self._corsika_configuration_debugging_parameters()
        config["IACT_PARAMETERS"] = self._corsika_configuration_iact_parameters(parameters_from_db)

        return config

    def _corsika_configuration_from_user_input(self, args_dict):
        """
        Get CORSIKA configuration from user input.

        Parameters
        ----------
        args_dict : dict
            Configuration dictionary.

        Returns
        -------
        dict
            Dictionary with CORSIKA parameters.
        """
        return {
            "EVTNR": [args_dict["event_number_first_shower"]],
            "NSHOW": [args_dict["nshow"]],
            "PRMPAR": [self.primary_particle.corsika7_id],
            "ESLOPE": [args_dict["eslope"]],
            "ERANGE": [
                args_dict["energy_range"][0].to("GeV").value,
                args_dict["energy_range"][1].to("GeV").value,
            ],
            "THETAP": [
                float(args_dict["zenith_angle"].to("deg").value),
                float(args_dict["zenith_angle"].to("deg").value),
            ],
            "PHIP": [
                self._rotate_azimuth_by_180deg(
                    args_dict["azimuth_angle"].to("deg").value,
                    correct_for_geomagnetic_field_alignment=args_dict[
                        "correct_for_b_field_alignment"
                    ],
                ),
                self._rotate_azimuth_by_180deg(
                    args_dict["azimuth_angle"].to("deg").value,
                    correct_for_geomagnetic_field_alignment=args_dict[
                        "correct_for_b_field_alignment"
                    ],
                ),
            ],
            "VIEWCONE": [
                args_dict["view_cone"][0].to("deg").value,
                args_dict["view_cone"][1].to("deg").value,
            ],
            "CSCAT": [
                args_dict["core_scatter"][0],
                args_dict["core_scatter"][1].to("cm").value,
                0.0,
            ],
        }

    def _corsika_configuration_interaction_flags(self, parameters_from_db):
        """
        Return CORSIKA interaction flags / parameters.

        Parameters
        ----------
        parameters_from_db : dict
            CORSIKA parameters from the database.

        Returns
        -------
        interaction_parameters : dict
            Dictionary with CORSIKA interaction parameters.
        """
        parameters = {}
        parameters["FIXHEI"] = self._input_config_first_interaction_height(
            parameters_from_db["corsika_first_interaction_height"]
        )
        parameters["FIXCHI"] = [
            self._input_config_corsika_starting_grammage(
                parameters_from_db["corsika_starting_grammage"]
            )
        ]
        parameters["TSTART"] = ["T"]
        parameters["ECUTS"] = self._input_config_corsika_particle_kinetic_energy_cutoff(
            parameters_from_db["corsika_particle_kinetic_energy_cutoff"]
        )
        parameters["MUADDI"] = ["F"]
        parameters["MUMULT"] = ["T"]
        parameters["LONGI"] = self._input_config_corsika_longitudinal_parameters(
            parameters_from_db["corsika_longitudinal_shower_development"]
        )
        parameters["MAXPRT"] = ["10"]
        parameters["ECTMAP"] = ["1.e6"]

        self._logger.debug(f"Interaction parameters: {parameters}")
        return parameters

    def _input_config_first_interaction_height(self, entry):
        """Return FIXHEI parameter CORSIKA format."""
        return [f"{entry['value']*u.Unit(entry['unit']).to('cm'):.2f}", "0"]

    def _input_config_corsika_starting_grammage(self, entry):
        """Return FIXCHI parameter CORSIKA format."""
        return f"{entry['value']*u.Unit(entry['unit']).to('g/cm2')}"

    def _input_config_corsika_particle_kinetic_energy_cutoff(self, entry):
        """Return ECUTS parameter CORSIKA format."""
        e_cuts = entry["value"]
        return [
            f"{e_cuts[0]*u.Unit(entry['unit']).to('GeV')} "
            f"{e_cuts[1]*u.Unit(entry['unit']).to('GeV')} "
            f"{e_cuts[2]*u.Unit(entry['unit']).to('GeV')} "
            f"{e_cuts[3]*u.Unit(entry['unit']).to('GeV')}"
        ]

    def _input_config_corsika_longitudinal_parameters(self, entry):
        """Return LONGI parameter CORSIKA format."""
        return ["T", f"{entry['value']*u.Unit(entry['unit']).to('g/cm2')}", "F", "F"]

    def _corsika_configuration_cherenkov_parameters(self, parameters_from_db):
        """
        Return CORSIKA Cherenkov emission parameters.

        Parameters
        ----------
        parameters_from_db : dict
            CORSIKA parameters from the database.

        Returns
        -------
        dict
            Dictionary with CORSIKA Cherenkov emission parameters.
        """
        parameters = {}
        parameters["CERSIZ"] = [parameters_from_db["corsika_cherenkov_photon_bunch_size"]["value"]]
        parameters["CERFIL"] = "0"
        parameters["CWAVLG"] = self._input_config_corsika_cherenkov_wavelength(
            parameters_from_db["corsika_cherenkov_photon_wavelength_range"]
        )
        self._logger.debug(f"Cherenkov parameters: {parameters}")
        return parameters

    def _input_config_corsika_cherenkov_wavelength(self, entry):
        """Return CWAVLG parameter CORSIKA format."""
        wavelength_range = entry["value"]
        return [
            f"{wavelength_range[0]*u.Unit(entry['unit']).to('nm')}",
            f"{wavelength_range[1]*u.Unit(entry['unit']).to('nm')}",
        ]

    def _corsika_configuration_iact_parameters(self, parameters_from_db):
        """
        Return CORSIKA IACT parameters.

        Parameters
        ----------
        parameters_from_db : dict
            CORSIKA parameters from the database.

        Returns
        -------
        dict
            Dictionary with CORSIKA IACT parameters.
        """
        parameters = {}
        parameters["SPLIT_AUTO"] = [parameters_from_db["corsika_iact_split_auto"]["value"]]
        parameters["IO_BUFFER"] = [
            self._input_config_io_buff(parameters_from_db["corsika_iact_io_buffer"])
        ]
        parameters["MAX_BUNCHES"] = [parameters_from_db["corsika_iact_max_bunches"]["value"]]
        self._logger.debug(f"IACT parameters: {parameters}")
        return parameters

    def _corsika_configuration_debugging_parameters(self):
        """Return CORSIKA debugging output parameters."""
        return {
            "DEBUG": ["F", 6, "F", 1000000],
            "DATBAS": ["yes"],
            "DIRECT": ["./"],
            "PAROUT": ["F", "F"],
        }

    def _input_config_io_buff(self, entry):
        """Return IO_BUFFER parameter CORSIKA format (Byte or MB required)."""
        value = entry["value"] * u.Unit(entry["unit"]).to("Mbyte")
        # check if value is integer-like
        if value.is_integer():
            return f"{int(value)}MB"
        return f"{int(entry['value'] * u.Unit(entry['unit']).to('byte'))}"

    def _rotate_azimuth_by_180deg(self, az, correct_for_geomagnetic_field_alignment=True):
        """
        Convert azimuth angle to the CORSIKA coordinate system.

        Parameters
        ----------
        az: float
            Azimuth angle in degrees.
        correct_for_geomagnetic_field_alignment: bool
            Whether to correct for the geomagnetic field alignment.

        Returns
        -------
        float
            Azimuth angle in degrees in the CORSIKA coordinate system.
        """
        b_field_declination = 0
        if correct_for_geomagnetic_field_alignment:
            b_field_declination = self.array_model.site_model.get_parameter_value("geomag_rotation")
        return (az + 180 + b_field_declination) % 360

    @property
    def primary(self):
        """Primary particle name."""
        return self.primary_particle.name

    def _set_primary_particle(self, args_dict):
        """
        Set primary particle from input dictionary.

        Parameters
        ----------
        args_dict: dict
            Input dictionary.

        Returns
        -------
        PrimaryParticle
            Primary particle.

        """
        if not args_dict or args_dict.get("primary_id_type") is None:
            return PrimaryParticle()
        return PrimaryParticle(
            particle_id_type=args_dict.get("primary_id_type"), particle_id=args_dict.get("primary")
        )

    def get_config_parameter(self, par_name):
        """
        Get value of CORSIKA configuration parameter.

        Parameters
        ----------
        par_name: str
            Name of the parameter as used in the CORSIKA input file (e.g. PRMPAR, THETAP ...).

        Raises
        ------
        KeyError
            When par_name is not a valid parameter name.

        Returns
        -------
        list
            Value(s) of the parameter.
        """
        par_value = []
        for _, values in self.config.items():
            if par_name in values:
                par_value = values[par_name]
        if len(par_value) == 0:
            self._logger.error(f"Parameter {par_name} is not a CORSIKA config parameter")
            raise KeyError
        return par_value if len(par_value) > 1 else par_value[0]

    def print_config_parameter(self):
        """Print CORSIKA config parameters for inspection."""
        for parameter_type, parameter_dict in self.config.items():
            print(f"Parameter type: {parameter_type}\n")
            for par, value in parameter_dict.items():
                print(f"{par} = {value}")

    @staticmethod
    def _get_text_single_line(pars, line_begin=""):
        """
        Return one parameter per line for each input parameter.

        Parameters
        ----------
        pars: dict
            Dictionary with the parameters to be written in the file.

        Returns
        -------
        str
            Text with the parameters.
        """
        text = ""
        for par, values in pars.items():
            line = line_begin + par + " "
            for v in values:
                line += str(v) + " "
            line += "\n"
            text += line
        return text

    def generate_corsika_input_file(self, use_multipipe=False, use_test_seeds=False):
        """
        Generate a CORSIKA input file.

        Parameters
        ----------
        use_multipipe: bool
            Whether to set the CORSIKA Inputs file to pipe
            the output directly to sim_telarray.

        """
        if self._is_file_updated:
            self._logger.debug(f"CORSIKA input file already updated: {self.config_file_path}")
            return self.config_file_path
        _output_generic_file_name = self.set_output_file_and_directory(use_multipipe=use_multipipe)
        self._logger.info(f"Exporting CORSIKA input file to {self.config_file_path}")

        with open(self.config_file_path, "w", encoding="utf-8") as file:
            file.write("\n* [ RUN PARAMETERS ]\n")
            text_parameters = self._get_text_single_line(self.config["USER_INPUT"])
            file.write(text_parameters)

            file.write("\n* [ SITE PARAMETERS ]\n")
            text_site_parameters = self._get_text_single_line(
                self.array_model.site_model.get_corsika_site_parameters(
                    config_file_style=True, model_directory=self.array_model.get_config_directory()
                )
            )
            file.write(text_site_parameters)

            file.write("\n* [ IACT ENV PARAMETERS ]\n")
            file.write(f"IACT setenv PRMNAME {self.primary_particle.name}\n")
            file.write(f"IACT setenv ZA {int(self.get_config_parameter('THETAP')[0])}\n")
            file.write(f"IACT setenv AZM {self.azimuth_angle}\n")

            file.write("\n* [ SEEDS ]\n")
            self._write_seeds(file, use_test_seeds)

            file.write("\n* [ TELESCOPES ]\n")
            telescope_list_text = self.get_corsika_telescope_list()
            file.write(telescope_list_text)

            file.write("\n* [ INTERACTION FLAGS ]\n")
            text_interaction_flags = self._get_text_single_line(self.config["INTERACTION_FLAGS"])
            file.write(text_interaction_flags)

            file.write("\n* [ CHERENKOV EMISSION PARAMETERS ]\n")
            text_cherenkov = self._get_text_single_line(
                self.config["CHERENKOV_EMISSION_PARAMETERS"]
            )
            file.write(text_cherenkov)

            file.write("\n* [ DEBUGGING OUTPUT PARAMETERS ]\n")
            text_debugging = self._get_text_single_line(self.config["DEBUGGING_OUTPUT_PARAMETERS"])
            file.write(text_debugging)

            file.write("\n* [ OUTPUT FILE ]\n")
            if use_multipipe:
                run_cta_script = Path(self.config_file_path.parent).joinpath("run_cta_multipipe")
                file.write(f"TELFIL |{run_cta_script!s}\n")
            else:
                file.write(f"TELFIL {_output_generic_file_name}\n")

            file.write("\n* [ IACT TUNING PARAMETERS ]\n")
            text_iact = self._get_text_single_line(
                self.config["IACT_PARAMETERS"],
                "IACT ",
            )
            file.write(text_iact)
            file.write("\nEXIT")

        # Write out the atmospheric transmission file to the model directory.
        # This is done explicitly because it is not done "automatically" when CORSIKA is not piped
        # to sim_telarray.
        self.array_model.site_model.export_atmospheric_transmission_file(
            model_directory=self.array_model.get_config_directory()
        )

        self._is_file_updated = True
        return self.config_file_path

    def get_corsika_config_file_name(self, file_type, run_number=None):
        """
        Get a CORSIKA config style file name for various configuration file types.

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
                Return CORSIKA input file name for one specific run.
                This is the input file after being pre-processed by sim_telarray (pfp).
            for file_type="config":
                Return generic CORSIKA config input file name.
            for file_type="output_generic"
                Return generic file name for the TELFIL option in the CORSIKA inputs file.
            for file_type="multipipe"
                Return multipipe "file name" for the TELFIL option in the CORSIKA inputs file.

        Raises
        ------
        ValueError
            If file_type is unknown or if the run number is not given for file_type==config_tmp.
        """
        file_label = f"_{self.label}" if self.label is not None else ""

        _vc_low = self.get_config_parameter("VIEWCONE")[0]
        _vc_high = self.get_config_parameter("VIEWCONE")[1]
        view_cone = (
            f"_cone{int(_vc_low):d}-{int(_vc_high):d}" if _vc_low != 0 or _vc_high != 0 else ""
        )

        base_name = (
            f"{self.primary_particle.name}_{self.array_model.site}_{self.array_model.layout_name}_"
            f"za{int(self.get_config_parameter('THETAP')[0]):03}-"
            f"azm{self.azimuth_angle:03}deg"
            f"{view_cone}{file_label}"
        )

        if file_type == "config_tmp":
            if run_number is None:
                raise ValueError("Must provide a run number for a temporary CORSIKA config file")
            return f"corsika_config_run{run_number:06}_{base_name}.txt"
        if file_type == "config":
            return f"corsika_config_{base_name}.input"
        if file_type == "output_generic":
            # The XXXXXX will be replaced by the run number after the pfp step with sed
            return (
                f"runXXXXXX_{base_name}_{self.array_model.site}_"
                f"{self.array_model.layout_name}{file_label}.zst"
            )
        if file_type == "multipipe":
            return f"multi_cta-{self.array_model.site}-{self.array_model.layout_name}.cfg"

        raise ValueError(f"The requested file type ({file_type}) is unknown")

    def set_output_file_and_directory(self, use_multipipe=False):
        """
        Set output file names and directories.

        Parameters
        ----------
        use_multipipe: bool
            Whether to set the CORSIKA Inputs file to pipe
            the output directly to sim_telarray. Defines directory names.

        Returns
        -------
        str
            Output file name.
        """
        sub_dir = "corsika_simtel" if use_multipipe else "corsika"
        config_file_name = self.get_corsika_config_file_name(file_type="config")
        file_directory = self.io_handler.get_output_directory(label=self.label, sub_dir=sub_dir)
        self._logger.debug(f"Creating directory {file_directory}")
        file_directory.mkdir(parents=True, exist_ok=True)
        self.config_file_path = file_directory.joinpath(config_file_name)

        return self.get_corsika_config_file_name(file_type="output_generic")

    def _write_seeds(self, file, use_test_seeds=False):
        """
        Generate and write seeds in the CORSIKA input file.

        Parameters
        ----------
        file: stream
            File where the telescope positions will be written.
        """
        random_seed = self.get_config_parameter("PRMPAR") + self.run_number
        rng = np.random.default_rng(random_seed)
        corsika_seeds = [534, 220, 1104, 382]
        if not use_test_seeds:
            corsika_seeds = [int(rng.uniform(0, 1e7)) for _ in range(4)]
        for s in corsika_seeds:
            file.write(f"SEED {s} 0 0\n")

    def get_corsika_telescope_list(self):
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

    @property
    def run_number(self):
        """Set run number."""
        return self._run_number

    @run_number.setter
    def run_number(self, run_number):
        """
        Set run number and validate it.

        Parameters
        ----------
        run_number: int
            Run number.
        """
        self._run_number = self.validate_run_number(run_number)

    def validate_run_number(self, run_number):
        """
        Validate run number and return it.

        Return run number from configuration if None.

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        int
            Run number.

        Raises
        ------
        ValueError
            If run_number is not a valid value (e.g., < 1).
        """
        if run_number is None:
            return self.run_number
        if not float(run_number).is_integer() or run_number < 1 or run_number > 999999:
            msg = f"Invalid type of run number ({run_number}) - it must be an uint < 1000000."
            self._logger.error(msg)
            raise ValueError(msg)
        return run_number
