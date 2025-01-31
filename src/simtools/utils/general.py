"""General functions useful across different parts of the code."""

import copy
import json
import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import yaml

__all__ = [
    "InvalidConfigDataError",
    "change_dict_keys_case",
    "clear_default_sim_telarray_cfg_directories",
    "collect_data_from_file",
    "collect_final_lines",
    "collect_kwargs",
    "get_log_excerpt",
    "get_log_level_from_user",
    "remove_substring_recursively_from_dict",
    "set_default_kwargs",
    "sort_arrays",
]

_logger = logging.getLogger(__name__)


class InvalidConfigDataError(Exception):
    """Exception for invalid configuration data."""


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
    FileNotFoundError
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
        raise TypeError(msg) from exc
    except urllib.error.HTTPError as exc:
        msg = f"Failed to download file from {url}"
        _logger.error(msg)
        raise FileNotFoundError(msg) from exc

    _logger.debug(f"Downloaded file from {url}")
    return data


def collect_data_from_file(file_name, yaml_document=None):
    """
    Collect data from file based on its extension.

    Parameters
    ----------
    file_name: str
        Name of the yaml/json/ascii file.
    yaml_document: None, int
        Return list of yaml documents or a single document (for yaml files with several documents).

    Returns
    -------
    data: dict or list
        Data as dict or list.
    """
    if is_url(file_name):
        return collect_data_from_http(file_name)

    suffix = Path(file_name).suffix.lower()
    with open(file_name, encoding="utf-8") as file:
        if suffix == ".json":
            return json.load(file)
        if suffix == ".list":
            return [line.strip() for line in file.readlines()]
        if suffix in [".yml", ".yaml"]:
            return _collect_data_from_yaml_file(file, file_name, yaml_document)
    return None


def _collect_data_from_yaml_file(file, file_name, yaml_document):
    """Collect data from a yaml file."""
    try:
        return yaml.safe_load(file)
    except yaml.constructor.ConstructorError:
        return _load_yaml_using_astropy(file)
    except yaml.composer.ComposerError:
        pass
    file.seek(0)
    if yaml_document is None:
        return list(yaml.safe_load_all(file))
    try:
        return list(yaml.safe_load_all(file))[yaml_document]
    except IndexError as exc:
        raise InvalidConfigDataError(
            f"YAML file {file_name} does not contain {yaml_document} documents."
        ) from exc


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
    try:
        log_level_lower = log_level.lower()
    except AttributeError:
        log_level_lower = log_level
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


def program_is_executable(program):
    """
    Check if program exists and is executable.

    Follows https://stackoverflow.com/questions/377017/

    """
    program = Path(program)

    def is_exe(fpath):
        return fpath.is_file() and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        try:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = Path(path) / program
                if is_exe(exe_file):
                    return exe_file
        except KeyError:
            _logger.warning("PATH environment variable is not set.")
            return None

    return None


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


def find_file(name, loc):
    """
    Search for files inside of given directories, recursively, and return its full path.

    Parameters
    ----------
    name: str
        File name to be searched for.
    loc: Path or list of Path
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
    """Get the age of a file in seconds since the last modification."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"'{file_path}' does not exist or is not a file.")

    file_stats = Path(file_path).stat()
    modification_time = file_stats.st_mtime
    current_time = time.time()

    return (current_time - modification_time) / 60


def _process_dict_keys(input_dict, case_func):
    """
    Process dictionary keys recursively.

    Parameters
    ----------
    input_dict: dict
        Dictionary to be processed.
    case_func: function
        Function to change case of keys (e.g., str.lower, str.upper).

    Returns
    -------
    dict
        Processed dictionary with keys changed.
    """
    output_dict = {}
    for key, value in input_dict.items():
        processed_key = case_func(key)
        if isinstance(value, dict):
            output_dict[processed_key] = _process_dict_keys(value, case_func)
        elif isinstance(value, list):
            processed_list = [
                _process_dict_keys(item, case_func) if isinstance(item, dict) else item
                for item in value
            ]
            output_dict[processed_key] = processed_list
        else:
            output_dict[processed_key] = value
    return output_dict


def change_dict_keys_case(data_dict, lower_case=True):
    """
    Change keys of a dictionary to lower or upper case recursively.

    Parameters
    ----------
    data_dict: dict
        Dictionary to be converted.
    lower_case: bool
        Change keys to lower (upper) case if True (False).

    Returns
    -------
    dict
        Dictionary with keys converted to lower or upper case.
    """
    # Determine which case function to use
    case_func = str.lower if lower_case else str.upper

    try:
        return _process_dict_keys(data_dict, case_func)
    except AttributeError as exc:
        _logger.error(f"Input is not a proper dictionary: {data_dict}")
        raise AttributeError from exc


def remove_substring_recursively_from_dict(data_dict, substring="\n"):
    """
    Remove substrings from all strings in a dictionary.

    Recursively crawls through the dictionary This e.g., allows to remove all newline characters
    from a dictionary.

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
    """Sort arrays.

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


def _get_value_dtype(value):
    """
    Get the data type of the given value.

    Parameters
    ----------
        Value to determine the data type.

    Returns
    -------
    type:
        Data type of the value.
    """
    if isinstance(value, (list | np.ndarray)):
        value = np.array(value)
        return value.dtype

    return type(value)


def validate_data_type(reference_dtype, value=None, dtype=None, allow_subtypes=True):
    """
    Validate data type of value or type object against a reference data type.

    Allow to check for exact data type or allow subtypes (e.g. uint is accepted for int).
    Take into account 'file' type as used in the model parameter database.

    Parameters
    ----------
    reference_dtype: str
        Reference data type to be checked against.
    value: any, optional
        Value to be checked (if dtype is None).
    dtype: type, optional
        Type object to be checked (if value is None).
    allow_subtypes: bool, optional
        If True, allow subtypes to be accepted.

    Returns
    -------
    bool:
        True if the data type is valid.
    """
    if value is None and dtype is None:
        raise ValueError("Either value or dtype must be given.")

    if value is not None and dtype is None:
        dtype = _get_value_dtype(value)

    # Strict comparison
    if not allow_subtypes:
        return np.issubdtype(dtype, reference_dtype)

    # Allow any sub-type of integer or float for success
    if (np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, "object")) and reference_dtype in (
        "string",
        "str",
        "file",
    ):
        return True

    if np.issubdtype(dtype, np.bool_) and reference_dtype in ("boolean", "bool"):
        return True

    if np.issubdtype(dtype, np.integer) and (
        np.issubdtype(reference_dtype, np.integer) or np.issubdtype(reference_dtype, np.floating)
    ):
        return True

    if np.issubdtype(dtype, np.floating) and np.issubdtype(reference_dtype, np.floating):
        return True

    return False


def convert_list_to_string(data, comma_separated=False, shorten_list=False, collapse_list=False):
    """
    Convert arrays to string (if required).

    Parameters
    ----------
    data: object
        Object of data to convert (e.g., double or list)
    comma_separated: bool
        If True, returns elements as a comma-separated string (default is space-separated).
    shorten_list: bool
        If True and all elements in the list are identical, returns a summary string
        like "all: value".  This is useful to make the configuration files more readable.
    collapse_list: bool
        If True and all elements in the list are identical, returns a single value
        instead of the entire list.

    Returns
    -------
    object or str:
        Converted data as string (if required)

    """
    if data is None or not isinstance(data, list | np.ndarray):
        return data
    if shorten_list and len(data) > 10 and all(np.isclose(item, data[0]) for item in data):
        return f"all: {data[0]}"
    if collapse_list and len(sorted(set(data))) == 1:
        data = [data[0]]
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


def read_file_encoded_in_utf_or_latin(file_name):
    """
    Read a file encoded in UTF-8 or Latin-1.

    Parameters
    ----------
    file_name: str
        Name of the file to be read.

    Returns
    -------
    list
        List of lines read from the file.

    Raises
    ------
    UnicodeDecodeError
        If the file cannot be decoded using UTF-8 or Latin-1.
    """
    try:
        with open(file_name, encoding="utf-8") as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        logging.debug("Unable to decode file using UTF-8. Trying Latin-1.")
        try:
            with open(file_name, encoding="latin-1") as file:
                lines = file.readlines()
        except UnicodeDecodeError as exc:
            raise UnicodeDecodeError("Unable to decode file using UTF-8 or Latin-1.") from exc

    return lines


def get_structure_array_from_table(table, column_names):
    """
    Get a structured array from an astropy table for a selected list of columns.

    Parameters
    ----------
    table: astropy.table.Table
        Table to be converted.
    column_names: list
        List of column names to be included in the structured array.

    Returns
    -------
    numpy.ndarray
        Structured array containing the table data.
    """
    return np.array(
        list(zip(*[np.array(table[col]) for col in column_names])),
        dtype=[(col, np.array(table[col]).dtype) for col in column_names],
    )


def convert_keys_in_dict_to_lowercase(data):
    """
    Recursively convert all dictionary keys to lowercase.

    Parameters
    ----------
    data: dict
        Dictionary to be converted.

    Returns
    -------
    dict
        Dictionary with all keys converted to lowercase.
    """
    if isinstance(data, dict):
        return {k.lower(): convert_keys_in_dict_to_lowercase(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys_in_dict_to_lowercase(i) for i in data]
    return data


def clear_default_sim_telarray_cfg_directories(command):
    """Prefix the command to clear default sim_telarray configuration directories.

    Parameters
    ----------
    command: str
        Command to be prefixed.

    Returns
    -------
    str
        Prefixed command.

    """
    return f"SIM_TELARRAY_CONFIG_PATH='' {command}"
