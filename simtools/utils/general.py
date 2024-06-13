"""
General functions useful across different parts of the code.
"""

import copy
import json
import logging
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections import namedtuple
from pathlib import Path
from urllib.parse import urlparse

import astropy.units as u
import numpy as np
import yaml

__all__ = [
    "change_dict_keys_case",
    "collect_data_from_file_or_dict",
    "collect_final_lines",
    "collect_kwargs",
    "InvalidConfigDataError",
    "InvalidConfigEntryError",
    "MissingRequiredConfigEntryError",
    "UnableToIdentifyConfigEntryError",
    "get_log_level_from_user",
    "remove_substring_recursively_from_dict",
    "separate_args_and_config_data",
    "set_default_kwargs",
    "validate_config_data",
    "get_log_excerpt",
    "sort_arrays",
]

_logger = logging.getLogger(__name__)


class UnableToIdentifyConfigEntryError(Exception):
    """Exception for unable to identify configuration entry."""


class MissingRequiredConfigEntryError(Exception):
    """Exception for missing required configuration entry."""


class InvalidConfigEntryError(Exception):
    """Exception for invalid configuration entry."""


class InvalidConfigDataError(Exception):
    """Exception for invalid configuration data."""


def validate_config_data(config_data, parameters, ignore_unidentified=False):
    """
    Validate a generic config_data dict by using the info
    given by the parameters dict. The entries will be validated
    in terms of length, units and names.

    See ./tests/resources/test_parameters.yml for an example of the structure
    of the parameters dict.

    Parameters
    ----------
    config_data: dict
        Input config data.
    parameters: dict
        Parameter information necessary for validation.
    ignore_unidentified: bool
        If set to True, unidentified parameters provided in config_data are ignored
        and a debug message is printed. Otherwise, an unidentified parameter leads to an error.

    Raises
    ------
    UnableToIdentifyConfigEntryError
        When an entry in config_data cannot be identified among the parameters.
    MissingRequiredConfigEntryError
        When a parameter without default value is not given in config_data.
    InvalidConfigEntryError
        When an entry in config_data is invalid (wrong len, wrong unit, ...).

    Returns
    -------
    namedtuple:
        Containing the validated config data entries.
    """

    def _process_identified_entry(key_data, value_data):
        nonlocal is_identified
        for par_name, par_info in parameters.items():
            names = par_info.get("names", [])
            if key_data == par_name or key_data.lower() in map(str.lower, names):
                validated_value = _validate_and_convert_value(par_name, par_info, value_data)
                out_data[par_name] = validated_value
                is_identified = True
                return True
        return False

    def handle_unidentified_entry(key_data):
        msg = f"Entry {key_data} in config_data cannot be identified"
        if ignore_unidentified:
            _logger.debug(f"{msg}, ignoring.")
        else:
            _logger.error(f"{msg}, stopping.")
            raise UnableToIdentifyConfigEntryError(msg)

    def process_default_value(par_name, par_info):
        if "default" not in par_info:
            msg = f"Required entry in config_data {par_name} was not given."
            _logger.error(msg)
            raise MissingRequiredConfigEntryError(msg)

        default_value = par_info["default"]

        if default_value is None:
            out_data[par_name] = None
        else:
            if isinstance(default_value, dict):
                default_value = default_value["value"]

            if "unit" in par_info and not isinstance(default_value, u.Quantity):
                default_value *= u.Unit(par_info["unit"])

            validated_value = _validate_and_convert_value(par_name, par_info, default_value)
            out_data[par_name] = validated_value

    out_data = {}

    if config_data is None:
        config_data = {}

    for key_data, value_data in config_data.items():
        is_identified = _process_identified_entry(key_data, value_data)

        if not is_identified:
            handle_unidentified_entry(key_data)

    for par_name, par_info in parameters.items():
        if par_name in out_data:
            continue

        process_default_value(par_name, par_info)

    configuration_data = namedtuple("configuration_data", out_data)
    return configuration_data(**out_data)


def _validate_and_convert_value_without_units(value, value_keys, par_name, par_info):
    """
    Validate input user parameter for input values without units.

    Parameters
    ----------
    value: list
       list of user input values.
    value_keys: list
       list of keys if user input was a dict; otherwise None.
    par_name: str
       name of parameter.
    par_info: dict
        dictionary with parameter info.

    Returns
    -------
    list, dict
        validated and converted input data

    """

    _, undefined_length = _check_value_entry_length(value, par_name, par_info)

    # Checking if values have unit and raising error, if so.
    if all(isinstance(v, str) for v in value):
        # In case values are string, e.g. mirror_numbers = 'all'
        # This is needed otherwise the elif condition will break
        pass
    elif any(u.Quantity(v).unit != u.dimensionless_unscaled for v in value):
        msg = f"Config entry {par_name} should not have units"
        _logger.error(msg)
        raise InvalidConfigEntryError(msg)

    if value_keys:
        return dict(zip(value_keys, value))
    return value if len(value) > 1 or undefined_length else value[0]


def _check_value_entry_length(value, par_name, par_info):
    """
    Validate length of user input parameters

    Parameters
    ----------
    value: list
        list of user input values
    par_name: str
        name of parameter
    par_info: dict
        dictionary with parameter info

    Returns
    -------
    value_length: int
        length of input list
    undefined_length: bool
        state of input list

    """

    # Checking the entry length
    value_length = len(value)
    _logger.debug(f"Value len of {par_name}: {value_length}")
    undefined_length = False
    try:
        if par_info["len"] is None:
            undefined_length = True
        elif value_length != par_info["len"]:
            msg = f"Config entry with wrong len: {par_name}"
            _logger.error(msg)
            raise InvalidConfigEntryError(msg)
    except KeyError:
        _logger.error("Missing len entry in par_info")
        raise

    return value_length, undefined_length


def _validate_and_convert_value_with_units(value, value_keys, par_name, par_info):
    """
    Validate input user parameter for input values with units.

    Parameters
    ----------
    value: list
       list of user input values
    value_keys: list
       list of keys if user input was a dict; otherwise None
    par_name: str
       name of parameter

    Returns
    -------
    list, dict
        validated and converted input data

    """

    value_length, undefined_length = _check_value_entry_length(value, par_name, par_info)

    par_unit = copy_as_list(par_info["unit"])

    if undefined_length and len(par_unit) != 1:
        msg = f"Config entry with undefined length should have a single unit: {par_name}"
        _logger.error(msg)
        raise InvalidConfigEntryError(msg)
    if len(par_unit) == 1:
        par_unit *= value_length

    # Checking units and converting them, if needed.
    value_with_units = []
    for arg, unit in zip(value, par_unit):
        # In case a entry is None, None should be returned.
        if unit is None or arg is None:
            value_with_units.append(arg)
            continue

        # Converting strings to Quantity
        if isinstance(arg, str):
            arg = u.quantity.Quantity(arg)

        if not isinstance(arg, u.quantity.Quantity):
            msg = f"Config entry given without unit: {par_name}"
            _logger.error(msg)
            raise InvalidConfigEntryError(msg)
        if not arg.unit.is_equivalent(unit):
            msg = (
                f"Config entry given with wrong unit: {par_name}"
                f" (should be {unit}, is {arg.unit})"
            )
            _logger.error(msg)
            raise InvalidConfigEntryError(msg)
        value_with_units.append(arg.to(unit).value)

    if value_keys:
        return dict(zip(value_keys, value_with_units))

    return (
        value_with_units if len(value_with_units) > 1 or undefined_length else value_with_units[0]
    )


def _validate_and_convert_value(par_name, par_info, value_in):
    """
    Validate input user parameter and convert it to the right units, if needed.
    Returns the validated arguments in a list.
    """

    if isinstance(value_in, dict):
        value = [d for (k, d) in value_in.items()]
        value_keys = [k for (k, d) in value_in.items()]
    else:
        value = copy_as_list(value_in)
        value_keys = None

    if "unit" not in par_info.keys():
        return _validate_and_convert_value_without_units(value, value_keys, par_name, par_info)

    return _validate_and_convert_value_with_units(value, value_keys, par_name, par_info)


def join_url_or_path(url_or_path, *args):
    """
    Join URL or path with additional subdirectories and file.
    This is the equivalent to Path.join(), with extended functionality
    working also for URLs.

    Parameters
    ----------
    url_or_path: str or Path
        URL or path to be extended.
    args: list
        Additional arguments to be added to the URL or path.

    Returns
    -------
    str or Path
        Extended URL or path.

    """
    if "://" in str(url_or_path):
        return "/".join([url_or_path.rstrip("/"), *args])
    return Path(url_or_path).joinpath(*args)


def is_url(url):
    """
    Check if a string is a valid URL.

    Parameters
    ----------
    url: str
        String to be checked.

    Returns
    -------
    bool
        True if url is a valid URL.

    """

    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False


def collect_data_from_http(url):
    """
    Download yaml or json file from url and return it contents as dict.
    File is downloaded as a temporary file and deleted afterwards.

    Parameters
    ----------
    url: str
        URL of the yaml/json file.

    Returns
    -------
    dict
        Dictionary containing the file content.

    Raises
    ------
    TypeError
        If url is not a valid URL.
    urllib.error.HTTPError
        If downloading the yaml file fails.

    """

    try:
        with tempfile.NamedTemporaryFile(mode="w+t") as tmp_file:
            urllib.request.urlretrieve(url, tmp_file.name)
            if url.endswith("yml") or url.endswith("yaml"):
                try:
                    data = yaml.safe_load(tmp_file)
                except yaml.constructor.ConstructorError:
                    data = _load_yaml_using_astropy(tmp_file)
            elif url.endswith("json"):
                data = json.load(tmp_file)
            elif url.endswith("list"):
                lines = tmp_file.readlines()
                data = [line.strip() for line in lines]
            else:
                msg = f"File extension of {url} not supported (should be json or yaml)"
                _logger.error(msg)
                raise TypeError(msg)
    except TypeError as exc:
        msg = "Invalid url {url}"
        _logger.error(msg)
        raise InvalidConfigDataError(msg) from exc
    except urllib.error.HTTPError as exc:
        msg = f"Failed to download file from {url}"
        _logger.error(msg)
        raise InvalidConfigDataError(msg) from exc

    _logger.debug(f"Downloaded file from {url}")
    return data


def collect_data_from_file_or_dict(file_name, in_dict, allow_empty=False):
    """
    Collect input data from file or dictionary.

    Parameters
    ----------
    file_name: str
        Name of the yaml/json/ascii file.
    in_dict: dict
        Data as dict.
    allow_empty: bool
        If True, an error won't be raised in case both file_name and dict are None.

    Returns
    -------
    data: dict or list
        Data as dict or list.
    """
    if file_name is not None:
        return collect_data_from_file(file_name, in_dict)

    if in_dict is not None:
        return dict(in_dict)

    if allow_empty:
        _logger.debug("Input has not been provided (neither by file, nor by dict)")
        return None

    msg = "Input has not been provided (neither by file, nor by dict)"
    _logger.debug(msg)
    raise InvalidConfigDataError(msg)


def collect_data_from_file(file_name, in_dict):
    """
    Collect data from file based on its extension.

    Parameters
    ----------
    file_name: str
        Name of the yaml/json/ascii file.
    in_dict: dict
        Data as dict.

    Returns
    -------
    data: dict or list
        Data as dict or list.
    """
    if in_dict is not None:
        _logger.warning("Both in_dict and file_name were given - file_name will be used")

    if is_url(file_name):
        return collect_data_from_http(file_name)

    with open(file_name, encoding="utf-8") as file:
        if Path(file_name).suffix.lower() == ".json":
            return json.load(file)

        if Path(file_name).suffix.lower() == ".list":
            lines = file.readlines()
            return [line.strip() for line in lines]

        try:
            return yaml.safe_load(file)
        except yaml.constructor.ConstructorError:
            return _load_yaml_using_astropy(file)


def collect_kwargs(label, in_kwargs):
    """
    Collect kwargs of the type label_* and return them as a dict.

    Parameters
    ----------
    label: str
        Label to be collected in kwargs.
    in_kwargs: dict
        kwargs.
    Returns
    -------
    dict
        Dictionary with the collected kwargs.
    """
    out_kwargs = {}
    for key, value in in_kwargs.items():
        if label + "_" in key:
            out_kwargs[key.replace(label + "_", "")] = value
    return out_kwargs


def set_default_kwargs(in_kwargs, **kwargs):
    """
    Fill in a dict with a set of default kwargs and return it.

    Parameters
    ----------
    in_kwargs: dict
        Input dict to be filled in with the default kwargs.
    **kwargs:
        Default kwargs to be set.

    Returns
    -------
    dict
        Dictionary containing the default kwargs.
    """
    for par, value in kwargs.items():
        if par not in in_kwargs.keys():
            in_kwargs[par] = value
    return in_kwargs


def collect_final_lines(file, n_lines):
    """
    Collect final lines.

    Parameters
    ----------
    file: str or Path
        File to collect the lines from.
    n_lines: int
        Number of lines to be collected.

    Returns
    -------
    str
        Final lines collected.
    """
    list_of_lines = []

    if Path(file).suffix == ".gz":
        import gzip  # pylint: disable=import-outside-toplevel

        file_open_function = gzip.open
    else:
        file_open_function = open
    with file_open_function(file, "rb") as read_obj:
        # Move the cursor to the end of the file
        read_obj.seek(0, os.SEEK_END)
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Get the current position of pointer i.e eof
        pointer_location = read_obj.tell()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            read_obj.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location - 1
            # read that byte / character
            new_byte = read_obj.read(1)
            # If the read byte is new line character then it means one line is read
            if new_byte == b"\n":
                # Save the line in list of lines
                list_of_lines.append(buffer.decode()[::-1])
                # If the size of list reaches n_lines, then return the reversed list
                if len(list_of_lines) == n_lines:
                    return "".join(list(reversed(list_of_lines)))
                # Reinitialize the byte array to save next line
                buffer = bytearray()
            else:
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)
        # As file is read completely, if there is still data in buffer, then its first line.
        if len(buffer) > 0:
            list_of_lines.append(buffer.decode()[::-1])

    return "".join(list(reversed(list_of_lines)))


def get_log_level_from_user(log_level):
    """
    Map between logging level from the user to logging levels of the logging module.

    Parameters
    ----------
    log_level: str
        Log level from the user.

    Returns
    -------
    logging.LEVEL
        The requested logging level to be used as input to logging.setLevel().
    """

    possible_levels = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level_lower = log_level.lower()
    if log_level_lower not in possible_levels:
        raise ValueError(
            f"'{log_level}' is not a logging level, "
            f"only possible ones are {list(possible_levels.keys())}"
        )

    return possible_levels[log_level_lower]


def copy_as_list(value):
    """
    Copy value and, if it is not a list, turn it into a list with a single entry.

    Parameters
    ----------
    value single variable of any type or list

    Returns
    -------
    value: list
        Copy of value if it is a list of [value] otherwise.
    """
    if isinstance(value, str):
        return [value]

    try:
        return list(value)
    except TypeError:
        return [value]


def separate_args_and_config_data(expected_args, **kwargs):
    """
    Separate kwargs into the arguments expected for instancing a class and the dict to be given as
    config_data. This function is specific for methods from_kwargs in classes which use the
    validate_config_data system.

    Parameters
    ----------
    expected_args: list of str
        List of arguments expected for the class.
    **kwargs

    Returns
    -------
    dict, dict
        A dict with the args collected and another one with config_data.
    """
    args = {}
    config_data = {}
    for key, value in kwargs.items():
        if key in expected_args:
            args[key] = value
        else:
            config_data[key] = value

    return args, config_data


def program_is_executable(program):
    """
    Checks if program exists and is executable

    Follows https://stackoverflow.com/questions/377017/

    """

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        try:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file
        except KeyError:
            _logger.debug("PATH environment variable is not set.")
            return None

    return None


def find_file(name, loc):
    """
    Search for files inside of given directories, recursively, and return its full path.

    Parameters
    ----------
    name: str
        File name to be searched for.
    loc: Path
        Location of where to search for the file.

    Returns
    -------
    Path
        Full path of the file to be found if existing. Otherwise, None.

    Raises
    ------
    FileNotFoundError
        If the desired file is not found.
    """

    all_locations = [loc] if not isinstance(loc, list) else loc

    def _search_directory(directory, filename, rec=False):
        if not Path(directory).exists():
            _logger.debug(f"Directory {directory} does not exist")
            return None

        file = Path(directory).joinpath(filename)
        if file.exists():
            _logger.debug(f"File {filename} found in {directory}")
            return file

        if rec:
            for subdir in Path(directory).iterdir():
                if subdir.is_dir():
                    file = _search_directory(subdir, filename, True)
                    if file:
                        return file
        return None

    # Searching file locally
    file = _search_directory(".", name)
    if file:
        return file

    # Searching file in given locations
    for location in all_locations:
        file = _search_directory(location, name, True)
        if file:
            return file

    msg = f"File {name} could not be found in {all_locations}"
    _logger.error(msg)
    raise FileNotFoundError(msg)


def get_log_excerpt(log_file, n_last_lines=30):
    """
    Get an excerpt from a log file, namely the n_last_lines of the file.

    Parameters
    ----------
    log_file: str or Path
        Log file to get the excerpt from.
    n_last_lines: int
        Number of last lines of the file to get.

    Returns
    -------
    str
        Excerpt from log file with header/footer
    """

    return (
        "\n\nRuntime error - See below the relevant part of the log/err file.\n\n"
        f"{log_file}\n"
        "====================================================================\n\n"
        f"{collect_final_lines(log_file, n_last_lines)}\n\n"
        "====================================================================\n"
    )


def get_file_age(file_path):
    """
    Get the age of a file in seconds since the last modification.
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"'{file_path}' does not exist or is not a file.")

    file_stats = os.stat(file_path)
    modification_time = file_stats.st_mtime
    current_time = time.time()

    file_age_minutes = (current_time - modification_time) / 60
    return file_age_minutes


def change_dict_keys_case(data_dict, lower_case=True):
    """
    Change keys of a dictionary to lower or upper case. Crawls through the dictionary and changes\
    all keys. Takes into account list of dictionaries, as e.g. found in the top level data model.

    Parameters
    ----------
    data_dict: dict
        Dictionary to be converted.
    lower_case: bool
        Change keys to lower (upper) case if True (False).
    """

    _return_dict = {}
    try:
        for key in data_dict.keys():
            if lower_case:
                _key_changed = key.lower()
            else:
                _key_changed = key.upper()
            if isinstance(data_dict[key], dict):
                _return_dict[_key_changed] = change_dict_keys_case(data_dict[key], lower_case)
            elif isinstance(data_dict[key], list):
                _tmp_list = []
                for _list_entry in data_dict[key]:
                    if isinstance(_list_entry, dict):
                        _tmp_list.append(change_dict_keys_case(_list_entry, lower_case))
                    else:
                        _tmp_list.append(_list_entry)
                _return_dict[_key_changed] = _tmp_list
            else:
                _return_dict[_key_changed] = data_dict[key]
    except AttributeError:
        _logger.error(f"Input is not a proper dictionary: {data_dict}")
        raise
    return _return_dict


def remove_substring_recursively_from_dict(data_dict, substring="\n"):
    """
    Remove substrings from all strings in a dictionary. Recursively crawls through the dictionary
    This e.g., allows to remove all newline characters from a dictionary.

    Parameters
    ----------
    data_dict: dict
        Dictionary to be converted.
    substring: str
        Substring to be removed.

    Raises
    ------
    AttributeError:
        if input is not a proper dictionary.
    """
    try:
        for key, value in data_dict.items():
            if isinstance(value, str):
                data_dict[key] = value.replace(substring, "")
            elif isinstance(value, list):
                modified_items = [
                    item.replace(substring, "") if isinstance(item, str) else item for item in value
                ]
                modified_items = [
                    (
                        remove_substring_recursively_from_dict(item, substring)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in modified_items
                ]
                data_dict[key] = modified_items
            elif isinstance(value, dict):
                data_dict[key] = remove_substring_recursively_from_dict(value, substring)
    except AttributeError:
        _logger.debug(f"Input is not a dictionary: {data_dict}")
    return data_dict


def sort_arrays(*args):
    """Sort arrays

    Parameters
    ----------
    *args
        Arguments to be sorted.
    Returns
    -------
    list
        Sorted args.
    """

    if len(args) == 0:
        return args
    order_array = copy.copy(args[0])
    new_args = []
    for arg in args:
        _, value = zip(*sorted(zip(order_array, arg)))
        new_args.append(list(value))
    return new_args


def extract_type_of_value(value) -> str:
    """
    Extract the string representation of the the type of a value.
    For example, for a string, it returns 'str' rather than '<class 'str'>'.
    Take into account also the case where the value is a numpy type.
    """
    _type = str(type(value))
    if "numpy" in _type:
        return re.sub(r"\d+", "", _type.split("'")[1].split(".")[-1])
    if "astropy" in _type:
        raise NotImplementedError("Astropy types are not supported yet.")

    _type = _type.split("'")[1]
    return _type


def get_value_unit_type(value, unit_str=None):
    """
    Get the value, unit and type of a value.
    The value is stripped of its unit and the unit is returned
    in its string form (i.e., to_string()).
    The type is returned as a string representation of the type.
    For example, for a string, it returns 'str' rather than '<class 'str'>'.
    An additional unit string can be given and the return value is converted to this units.

    Note that Quantities are always floats, even if the original value is represented as an int.

    Parameters
    ----------
    value: str, int, float, bool, u.Quantity
        Value to be parsed.
    unit_str: str
        Unit to be used for the value.

    Returns
    -------
    type of value, str, str
        Value, unit in string representation (to_string())),
        and string representation of the type of the value.
    """

    base_unit = None
    if isinstance(value, str | u.Quantity):
        try:
            _quantity_value = u.Quantity(value)
            base_value = _quantity_value.value
            base_type = extract_type_of_value(base_value)
            if _quantity_value.unit.to_string() != "":
                base_unit = _quantity_value.unit.to_string()
                try:  # handle case of e.g., "0 0" and avoid unit.scale
                    float(base_unit)
                    base_value = value
                    base_type = "str"
                    base_unit = None
                except ValueError:
                    pass
        # ValueError: covers strings of type "5 not a unit"
        except (TypeError, ValueError):
            base_value = value
            base_type = "str"
    else:
        base_value = value
        base_type = extract_type_of_value(base_value)

    if unit_str is not None:
        try:
            base_value = base_value * u.Unit(base_unit).to(u.Unit(unit_str))
        except u.UnitConversionError:
            _logger.error(f"Cannot convert {base_unit} to {unit_str}.")
            raise
        except TypeError:
            pass
        base_unit = unit_str

    return base_value, base_unit, base_type


def get_value_as_quantity(value, unit):
    """
    Get a value as a Quantity with a given unit. If value is a Quantity, convert to the given unit.

    Parameters
    ----------
    value:
        value to get a unit. It can be a float, int, or a Quantity (convertible to 'unit').
    unit: astropy.units.Unit
        Unit to apply to 'quantity'.

    Returns
    -------
    astropy.units.Quantity
        Quantity of value 'quantity' and unit 'unit'.

    Raises
    ------
    u.UnitConversionError
        If the value cannot be converted to the given unit.
    """
    if isinstance(value, u.Quantity):
        try:
            value = value.to(unit)
            return value
        except u.UnitConversionError:
            _logger.error(f"Cannot convert {value.unit} to {unit}.")
            raise
    return value * unit


def user_confirm():
    """
    Ask the user to enter y or n (case-insensitive) on the command line.

    Returns
    -------
    bool:
        True if the answer is Y/y.

    """
    while True:
        try:
            answer = input("Is this OK? [y/n]").lower()
            return answer == "y"
        except EOFError:
            break
    return False


def validate_data_type(reference_dtype, value=None, dtype=None, allow_subtypes=True):
    """
    Validate data type of value (scalar, list, np array) or type object against a
    reference data type. Allow to check for exact data type or allow sub types
    (e.g. uint is accepted for int).  Take into account 'file' type as used in the
    model parameter database

    Parameters
    ----------
    reference_dtype: str
        Reference data type to be checked against.
    value: any
        Value to be checked (if dtype is None).
    dtype: type
        Type object to be checked (if value is None).
    allow_subtypes: bool
        If True, allow subtypes to be accepted.

    Returns
    -------
    bool:
        True if the data type is valid.

    """
    if value is not None and dtype is None:
        if isinstance(value, list | np.ndarray):
            value = np.array(value)
            dtype = value.dtype
        else:
            dtype = type(value)
    elif value is None and dtype is None:
        raise ValueError("Either value or dtype must be given.")

    # strict comparison
    if not allow_subtypes:
        return np.issubdtype(dtype, reference_dtype)
    # allow any sub-type of integer or float for success
    # dtype is 'object' for 'file' type and value None
    if (np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, "object")) and reference_dtype in (
        "string",
        "str",
        "file",
    ):
        return True
    if np.issubdtype(dtype, np.bool_) and reference_dtype in ("boolean", "bool"):
        return True
    # allow ints to be converted to floats
    if np.issubdtype(dtype, np.integer) and (
        np.issubdtype(reference_dtype, np.integer) or np.issubdtype(reference_dtype, np.floating)
    ):
        return True
    if np.issubdtype(dtype, np.floating) and np.issubdtype(reference_dtype, np.floating):
        return True

    return False


def convert_list_to_string(data, comma_separated=False):
    """
    Convert arrays to string (if required)

    Parameters
    ----------
    data: object
        Object of data to convert (e.g., double or list)
    comma_separated: bool
        If True, return arrays as comma separated strings.

    Returns
    -------
    object or str:
        Converted data as string (if required)

    """
    if data is None or not isinstance(data, list | np.ndarray):
        return data
    if comma_separated:
        return ", ".join(str(item) for item in data)
    return " ".join(str(item) for item in data)


def convert_string_to_list(data_string, is_float=True):
    """
    Convert string (as used e.g. in sim_telarray) to list.
    Allow coma or space separated strings.

    Parameters
    ----------
    data_string: object
        String to be converted

    Returns
    -------
    list, str
        Converted data from string (if required).
        Return data_string if conversion fails.

    """

    try:
        if is_float:
            return [float(v) for v in data_string.split()]
        return [int(v) for v in data_string.split()]
    except ValueError:
        pass
    if "," in data_string:
        result = data_string.split(",")
        return [item.strip() for item in result]
    if " " in data_string:
        return data_string.split()
    return data_string


def _load_yaml_using_astropy(file):
    """
    Load a yaml file using astropy's yaml loader.

    Parameters
    ----------
    file: file
        File to be loaded.

    Returns
    -------
    dict
        Dictionary containing the file content.
    """
    # pylint: disable=import-outside-toplevel
    import astropy.io.misc.yaml as astropy_yaml

    file.seek(0)
    return astropy_yaml.load(file)
