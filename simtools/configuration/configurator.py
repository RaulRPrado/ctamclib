import logging
import os
import sys

import yaml

import simtools.configuration.commandline_parser as argparser
from simtools import io_handler

__all__ = [
    "Configurator",
    "InvalidConfigurationParameter",
]


class InvalidConfigurationParameter(Exception):
    """Exception for Invalid configuration parameter."""

    pass


class Configurator:
    """
    Configuration handling application configuration.

    Allow to set configuration parameters by

    - command line arguments
    - configuration file (yml file)
    - configuration dict when calling the class
    - environmental variables

    Configuration parameter names are converted always to lower case.

    Parameters
    ----------
    config: (dict, optional)
        Configuration parameters as dict (default is None).
    label: (str, optional)
        Class label (default is None).
    usage: (str, opitonal)
        Application usage description (default is None).
    description: (str, optional)
        Text displayed as description (default is None).
    epilog: (str, optional)
        Text display after all arguments (default is None).
    """

    def __init__(self, config=None, label=None, usage=None, description=None, epilog=None):
        """
        Initialize Configurator.
        """

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init Configuration")

        self.config_class_init = config
        self.label = label
        self.config = {}
        self.parser = argparser.CommandLineParser(
            prog=self.label, usage=usage, description=description, epilog=epilog
        )

    def default_config(self, arg_list=None, add_db_config=False):
        """
        Returns dictionary of default configuration

        Parameters
        ----------
        arg_list (list, optional)
            List of arguments (default is None)
        add_db_config (bool, optional)
            Add DB configuration file (default is False)

        Returns
        -------
        dict
            Configuration parameters as dict.
        """

        self.parser.initialize_default_arguments()
        if arg_list and "--site" in arg_list:
            self.parser.initialize_telescope_model_arguments(True, "--telescope" in arg_list)
        if add_db_config:
            self.parser.initialize_db_config_arguments()

        self._fill_config(arg_list)
        return self.config

    def initialize(
        self,
        paths=True,
        telescope_model=False,
        workflow_config=False,
        db_config=False,
        job_submission=False,
    ):
        """
        Initialize configuration from command line, configuration file, class config, or \
         environmental variable.

        Priorities in parameter settings.
        1. command line; 2. yaml file; 3. class init; 4. env variables.

        Conflicting configuration settings raise an Exception, with the exception of settings \
        from environmental variables, which are only done when the configuration parameter \
        is None.

        Parameters
        ----------
        paths: (bool, optional)
            Add path configuration to list of args (default is True).
        telescope_model: (bool, optional)
            Add telescope model configuration to list of args (default is False).
        workflow_config: (bool, optional)
            Add workflow configuration to list of args (default is False).
        db_config: (bool, optional)
            Add database configuration parameters to list of args (default is False).
        job_submission: (bool, optional)
            Add job submission configuration to list of args (default is False).

        Returns
        -------
        dict
            Configuration parameters as dict.
        dict
            Dictionary with DB parameters

        Raises
        ------
        InvalidConfigurationParameter
           if parameter has already been defined with a different value.

        """

        self.parser.initialize_default_arguments(
            paths=paths,
            telescope_model=telescope_model,
            workflow_config=workflow_config,
            db_config=db_config,
            job_submission=job_submission,
        )

        self._fill_from_command_line()
        try:
            self._fill_from_config_file(self.config["workflow_config"])
        except KeyError:
            pass
        try:
            self._fill_from_config_file(self.config["config"])
        except KeyError:
            pass
        self._fill_from_config_dict(self.config_class_init)
        self._fill_from_environmental_variables()
        self._initialize_io_handler()
        _db_dict = self._get_db_parameters()

        if self.config["label"] is None:
            self.config["label"] = self.label

        return self.config, _db_dict

    def _fill_from_command_line(self, arg_list=None):
        """
        Fill configuration parameters from command line arguments.

        """

        if arg_list is None:
            arg_list = sys.argv[1:]

        self._fill_config(arg_list)

    def _fill_from_config_dict(self, _input_dict):
        """
        Fill configuration parameters from dictionary. Enforce that configuration parameter names\
         are lower case.

        Parameters
        ----------
        _input_dict: dict
            dictionary with configuration parameters.

        """
        _tmp_config = {}
        try:
            for key, value in _input_dict.items():
                self._check_parameter_configuration_status(key, value)
                _tmp_config[key.lower()] = value
        except AttributeError:
            pass

        self._fill_config(_tmp_config)

    def _check_parameter_configuration_status(self, key, value):
        """
        Check if a parameter is already configured and not still set to the default value. Allow \
        configuration with None values.

        Parameters
        ----------
        key, value
           parameter key, value to be checked


        Raises
        ------
        InvalidConfigurationParameter
           if parameter has already been defined with a different value.


        """
        # parameter not changed or None
        if self.parser.get_default(key) == self.config[key] or self.config[key] is None:
            return

        # parameter already set
        if key in self.config and self.config[key] != value:
            self._logger.error(
                "Inconsistent configuration parameter ({}) definition ({} vs {})".format(
                    key, self.config[key], value
                )
            )
            raise InvalidConfigurationParameter

    def _fill_from_config_file(self, config_file):
        """
        Read and fill configuration parameters from yaml file. Take into account that this could be\
        a CTASIMPIPE workflow configuration file. (CTASIMPIPE:CONFIGURATION is optional, therefore,\
        no error is raised when this key is not found)

        Parameters
        ----------
        config file: str
            Name of configuration file name


        Raises
        ------
        FileNotFoundError
           if configuration file has not been found.

        """

        try:
            self._logger.debug("Reading configuration from {}".format(config_file))
            with open(config_file, "r") as stream:
                _config_dict = yaml.safe_load(stream)
            if "CTASIMPIPE" in _config_dict:
                try:
                    self._fill_from_config_dict(_config_dict["CTASIMPIPE"]["CONFIGURATION"])
                except KeyError:
                    self._logger.info(
                        "No CTASIMPIPE:CONFIGURATION dict found in {}.".format(config_file)
                    )
            else:
                self._fill_from_config_dict(_config_dict)
        # TypeError is raised for config_file=None
        except TypeError:
            pass
        except FileNotFoundError:
            self._logger.error("Configuration file not found: {}".format(config_file))
            raise

    def _fill_from_environmental_variables(self):
        """
        Fill any unconfigured configuration parameters (i.e., parameter is None) \
        from environmental variables.

        """

        _env_dict = {}
        try:
            for key, value in self.config.items():
                if value is None:
                    _env_dict[key] = os.environ.get(key.upper())
        except AttributeError:
            pass

        self._fill_from_config_dict(_env_dict)

    def _initialize_io_handler(self):
        """
        Initialize IOHandler with input and output paths.

        """
        _io_handler = io_handler.IOHandler()
        _io_handler.set_paths(
            output_path=self.config.get("output_path", None),
            data_path=self.config.get("data_path", None),
            model_path=self.config.get("model_path", None),
        )

    @staticmethod
    def _arglist_from_config(input_var):
        """
        Convert input list of strings as needed by argparse.

        Special cases:
        - lists as arguments (using e.g., nargs="+") are expanded
        - boolean are expected to be handled as action="store_true" or "store_false"
        - None values or zero length values are ignored (this means setting a parameter \
            to none or "" is not allowed.


        Ignore values which are None or of zero length.

        Parameters
        ----------
        input_var: dict, list, None
           Dictionary/list of commands to convert to list.

        Returns
        -------
        list
            Dict keys and values as dict.

        """

        if isinstance(input_var, dict):
            _list_args = []
            for key, value in input_var.items():
                if isinstance(value, list):
                    _list_args.append("--" + key)
                    _list_args += value
                elif not isinstance(value, bool) and value is not None and len(str(value)) > 0:
                    _list_args.append("--" + key)
                    _list_args.append(str(value))
                elif value:
                    _list_args.append("--" + key)
            return _list_args

        try:
            return [str(value) for value in list(input_var) if value != "None"]
        except TypeError:
            return []

    @staticmethod
    def _convert_stringnone_to_none(input_dict):
        """
        Convert string type 'None' to type None (argparse returns None as str).

        Parameters
        ----------
        input_dict
            Dictionary with values to be converted.

        """

        for key, value in input_dict.items():
            input_dict[key] = None if value == "None" else value

        return input_dict

    def _fill_config(self, input_container):
        """
        Fill configuration dictionary.

        Parameters
        ----------
        input_container
            List or dictionary with configuration updates.

        """

        self.config = self._convert_stringnone_to_none(
            vars(
                self.parser.parse_args(
                    self._arglist_from_config(self.config)
                    + self._arglist_from_config(input_container)
                )
            )
        )

    def _get_db_parameters(self):
        """
        Return parameters for DB configuration

        Parameters
        ----------
        dict
            Dictionary with DB parameters


        """

        _db_dict = {}
        _db_para = ("db_api_user", "db_api_pw", "db_api_port", "db_server")
        try:
            for _para in _db_para:
                _db_dict[_para] = self.config[_para]
        except KeyError:
            pass

        return _db_dict
