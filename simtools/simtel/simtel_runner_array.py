import logging
from pathlib import Path

import simtools.util.general as gen
from simtools import io_handler
from simtools.simtel.simtel_runner import InvalidOutputFile, SimtelRunner

__all__ = ["SimtelRunnerArray"]


class SimtelRunnerArray(SimtelRunner):
    """
    SimtelRunnerArray is the interface with sim_telarray to perform array simulations.

    Configurable parameters:
        simtel_data_directory:
            len: 1
            default: null
            unit: null
        primary:
            len: 1
            unit: null
        zenith_angle:
            len: 1
            unit: deg
            default: 20 deg
        azimuth_angle:
            len: 1
            unit: deg
            default: 0 deg

    Attributes
    ----------
    label: str, optional
        Instance label.
    array_model: ArrayModel
        Instance of the ArrayModel class.
    config: namedtuple
        Contains the configurable parameters (zenith_angle).

    Methods
    -------
    get_run_script(self, test=False, input_file=None, run_number=None)
        Builds and returns the full path of the bash run script containing
        the sim_telarray command.
    run(test=False, force=False, input_file=None, run_number=None)
        Run sim_telarray. test=True will make it faster and force=True will remove existing files
        and run again.
    get_info_for_file_name()
        Get a dirctionary with the info necessary for building the sim_telarray file names.
    get_file_name()
        Get a sim_telarray style file name for various file types.
    """

    def __init__(
        self,
        array_model,
        label=None,
        simtel_source_path=None,
        config_data=None,
        config_file=None,
    ):
        """
        SimtelRunnerArray.

        Parameters
        ----------
        array_model: str
            Instance of TelescopeModel class.
        label: str, optional
            Instance label. Important for output file naming.
        simtel_source_path: str (or Path), optional
            Location of sim_telarray installation.
        config_data: dict.
            Dict containing the configurable parameters.
        config_file: str or Path
            Path of the yaml file containing the configurable parameters.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelRunnerArray")

        super().__init__(label=label, simtel_source_path=simtel_source_path)

        self.array_model = self._validate_array_model(array_model)
        self.label = label if label is not None else self.array_model.label

        self.io_handler = io_handler.IOHandler()

        self._base_directory = self.io_handler.get_output_directory(self.label, "array-simulator")

        # Loading config_data
        _config_data_in = gen.collect_data_from_yaml_or_dict(config_file, config_data)
        _parameter_file = self.io_handler.get_input_data_file(
            "parameters", "simtel-runner-array_parameters.yml"
        )
        _parameters = gen.collect_data_from_yaml_or_dict(_parameter_file, None)
        self.config = gen.validate_config_data(_config_data_in, _parameters)

        self._load_simtel_data_directories()

    def _load_simtel_data_directories(self):
        """
        Create sim_telarray output directories for data, log and input.

        If simtel_data_directory is not given as a configurable parameter,
        the standard directory of simtools output (simtools-output) will
        be used. A sub directory simtel-data will be created and sub directories for
        log and data will be created inside it.
        """

        if self.config.simtel_data_directory is None:
            # Default config value
            simtel_base_dir = self._base_directory
        else:
            simtel_base_dir = Path(self.config.simtel_data_directory)

        simtel_base_dir = simtel_base_dir.joinpath("simtel-data")
        simtel_base_dir = simtel_base_dir.joinpath(self.array_model.site)
        simtel_base_dir = simtel_base_dir.joinpath(self.config.primary)
        simtel_base_dir = simtel_base_dir.absolute()

        self._simtel_data_dir = simtel_base_dir.joinpath("data")
        self._simtel_data_dir.mkdir(parents=True, exist_ok=True)
        self._simtel_log_dir = simtel_base_dir.joinpath("log")
        self._simtel_log_dir.mkdir(parents=True, exist_ok=True)

    def get_info_for_file_name(self, run_number):
        """
        Get a dirctionary with the info necessary for building the sim_telarray file names.

        Returns
        -------
        dict
            Dictionary with the keys necessary for building the sim_telarray file names.
        """
        return {
            "run": run_number,
            "primary": self.config.primary,
            "array_name": self.array_model.layout_name,
            "site": self.array_model.site,
            "zenith": self.config.zenith_angle,
            "azimuth": self.config.azimuth_angle,
            "label": self.label,
        }

    def get_file_name(self, file_type, **kwargs):
        """
        Get a sim_telarray style file name for various file types.

        Parameters
        ----------
        file_type: str
            The type of file it is (determines the file suffix).
            Choices are log, histogram, output or sub_log.
        kwargs: dict
            The dictionary must include the following parameters (unless listed as optional):
                run: int
                    Run number.
                primary: str
                    Primary particle (e.g gamma, proton etc).
                zenith: float
                    Zenith angle (deg).
                azimuth: float
                    Azimuth angle (deg).
                site: str
                    Paranal or LaPalma.
                array_name: str
                    Array name.
                label: str
                    Instance label (optional).
                mode: str
                    out or err (optional, relevant only for sub_log).

        Returns
        -------
        str
            File name with full path.

        Raises
        ------
        ValueError
            If file_type is unknown.
        """

        file_label = (
            f"_{kwargs['label']}" if "label" in kwargs and kwargs["label"] is not None else ""
        )
        file_name = (
            f"run{kwargs['run']}_{kwargs['primary']}_"
            f"za{int(kwargs['zenith']):d}deg_azm{int(kwargs['azimuth']):d}deg_"
            f"{kwargs['site']}_{kwargs['array_name']}{file_label}"
        )
        if file_type == "log":
            file_name += ".log"
            return self._simtel_log_dir.joinpath(file_name)
        elif file_type == "histogram":
            file_name += ".hdata.zst"
            return self._simtel_data_dir.joinpath(file_name)
        elif file_type == "output":
            file_name += ".simtel.zst"
            return self._simtel_data_dir.joinpath(file_name)
        elif file_type == "sub_log":
            suffix = ".log"
            if "mode" in kwargs:
                suffix = f".{kwargs['mode']}"
            file_name = f"log_sub_{file_name}{suffix}"
            return self._simtel_log_dir.joinpath(file_name)
        else:
            raise ValueError(f"The requested file type ({file_type}) is unknown")

    def has_sub_log_file(self, run_number, mode="out"):
        """
        Checks that the sub run log file for this run number
        is a valid file on disk

        Parameters
        ----------
        run_number: int
            Run number.

        """

        info_for_file_name = self.get_info_for_file_name(run_number)
        run_sub_file = self.get_file_name("sub_log", **info_for_file_name, mode=mode)
        return Path(run_sub_file).is_file()

    def get_resources(self, run_number):
        """
        Reading run time from last line of submission log file.

        Parameters
        ----------
        run_number: int
            Run number.

        Returns
        -------
        dict
            run time of job in seconds

        """

        info_for_file_name = self.get_info_for_file_name(run_number)
        sub_log_file = self.get_file_name("sub_log", **info_for_file_name, mode="out")

        self._logger.debug("Reading resources from {}".format(sub_log_file))

        _resources = {}

        _resources["runtime"] = None
        with open(sub_log_file, "r") as file:
            for line in reversed(list(file)):
                if "RUNTIME" in line:
                    _resources["runtime"] = int(line.split()[1])
                    break

        if _resources["runtime"] is None:
            self._logger.debug("RUNTIME was not found in run log file")

        return _resources

    def _shall_run(self, run_number=None):
        """Tells if simulations should be run again based on the existence of output files."""
        output_file = self.get_file_name(
            file_type="output", **self.get_info_for_file_name(run_number)
        )
        return not output_file.exists()

    def _make_run_command(self, input_file, run_number=1):
        """
        Builds and returns the command to run simtel_array.

        Attributes
        ----------
        input_file: str
            Full path of the input CORSIKA file
        run_number: int
            run number

        """

        info_for_file_name = self.get_info_for_file_name(run_number)
        self._log_file = self.get_file_name(file_type="log", **info_for_file_name)
        histogram_file = self.get_file_name(file_type="histogram", **info_for_file_name)
        output_file = self.get_file_name(file_type="output", **info_for_file_name)

        # Array
        command = str(self._simtel_source_path.joinpath("sim_telarray/bin/sim_telarray"))
        command += " -c {}".format(self.array_model.get_config_file())
        command += " -I{}".format(self.array_model.get_config_directory())
        command += super()._config_option("telescope_theta", self.config.zenith_angle)
        command += super()._config_option("telescope_phi", self.config.azimuth_angle)
        command += super()._config_option("power_law", "2.5")
        command += super()._config_option("histogram_file", histogram_file)
        command += super()._config_option("output_file", output_file)
        command += super()._config_option("random_state", "auto")
        command += super()._config_option("show", "all")
        command += " " + str(input_file)
        command += " > " + str(self._log_file) + " 2>&1"

        return command

    def _check_run_result(self, run_number):
        # Checking run
        output_file = self.get_file_name(
            file_type="output", **self.get_info_for_file_name(run_number)
        )
        if not output_file.exists():
            msg = "sim_telarray output file does not exist."
            self._logger.error(msg)
            raise InvalidOutputFile(msg)
        else:
            self._logger.debug("Everything looks fine with the sim_telarray output file.")
