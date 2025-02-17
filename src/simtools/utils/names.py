"""Validation of names."""

import logging
import re
from functools import cache
from pathlib import Path

import yaml

from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH, SCHEMA_PATH

_logger = logging.getLogger(__name__)

__all__ = [
    "generate_file_name",
    "get_array_element_type_from_name",
    "get_site_from_array_element_name",
    "layout_telescope_list_file_name",
    "sanitize_name",
    "simtel_config_file_name",
    "simtel_single_mirror_list_file_name",
    "validate_array_element_id_name",
    "validate_array_element_name",
    "validate_site_name",
]


@cache
def array_elements():
    """
    Load array elements from reference files and keep in cache.

    Returns
    -------
    dict
        Array elements.
    """
    with open(Path(SCHEMA_PATH) / "array_elements.yml", encoding="utf-8") as file:
        return yaml.safe_load(file)["data"]


@cache
def site_names():
    """
    Site names from reference file.

    The list of sites is derived from the sites listed in the model parameter
    schema files. Return a dictionary for compatibility with the validation routines.

    Returns
    -------
    dict
        Site names.
    """
    _array_elements = array_elements()
    _sites = set()
    for entry in _array_elements.values():
        site = entry["site"]
        if isinstance(site, list):
            _sites.update(site)
        else:
            _sites.add(site)
    return {site: [site.lower()] for site in _sites}


@cache
def array_element_design_types(array_element_type):
    """
    Array element site types (e.g., 'design' or 'flashcam').

    Default value is ['design', 'test'].

    Parameters
    ----------
    array_element_type
        Array element type

    Returns
    -------
    list
        Array element design types.
    """
    default_types = ["design", "test"]
    if array_element_type is None:
        return default_types
    return array_elements()[array_element_type].get("design_types", default_types)


@cache
def load_model_parameters(class_key_list):
    model_parameters = {}
    schema_files = list(Path(MODEL_PARAMETER_SCHEMA_PATH).rglob("*.yml"))
    for schema_file in schema_files:
        with open(schema_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        try:
            if data["instrument"]["class"] in class_key_list:
                model_parameters[data["name"]] = data
        except KeyError:
            pass
    return model_parameters


def site_parameters():
    return load_model_parameters(class_key_list="Site")


def telescope_parameters():
    return load_model_parameters(class_key_list=("Structure", "Camera", "Telescope"))


def validate_array_element_id_name(name, array_element_type=None):
    """
    Validate array element ID.

    Allowed IDs are
    - design (for design array elements or testing)
    - array element ID (e.g., 1, 5, 15)
    - test (for testing)

    Parameters
    ----------
    name: str or int
        Array element ID name.
    array_element_type: str
        Array element type (e.g., LSTN, MSTN).

    Returns
    -------
    str
        Validated array element ID (added leading zeros, e.g., 1 is converted to 01).

    Raises
    ------
    ValueError
        If name is not valid.
    """
    if isinstance(name, int) or name.isdigit():
        return f"{int(name):02d}"
    if name.lower() in {t.lower() for t in array_element_design_types(array_element_type)}:
        return str(name)

    msg = f"Invalid array element ID name {name}"
    _logger.error(msg)
    raise ValueError(msg)


def validate_site_name(name):
    """
    Validate site name.

    Parameters
    ----------
    name: str
        Site name.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, site_names())


def _validate_name(name, all_names):
    """
    Validate name given the all_names options.

    For each key in all_names, a list of options is given.
    If name is in this list, the key name is returned.

    Parameters
    ----------
    name: str
        Name to validate.
    all_names: dict
        Dictionary with valid names.

    Returns
    -------
    str
        Validated name.

    Raises
    ------
    ValueError
        If name is not valid.
    """
    for key in all_names.keys():
        if isinstance(all_names[key], list) and name.lower() in [
            item.lower() for item in all_names[key]
        ]:
            return key
        if name.lower() == key.lower():
            return key

    msg = f"Invalid name {name}"
    raise ValueError(msg)


def validate_array_element_type(name):
    """
    Validate array element type (e.g., LSTN, MSTN).

    Parameters
    ----------
    name: str
        Array element type.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, array_elements())


def validate_array_element_name(name):
    """
    Validate array element name (e.g., MSTx-NectarCam, MSTN-01).

    Parameters
    ----------
    name: str
        Array element name.

    Returns
    -------
    str
        Validated name.
    """
    try:
        _array_element_type, _array_element_id = name.split("-")
    except ValueError as exc:
        msg = f"Invalid name {name}"
        raise ValueError(msg) from exc
    if _array_element_type == "OBS":
        return validate_site_name(_array_element_id)
    return (
        _validate_name(_array_element_type, array_elements())
        + "-"
        + validate_array_element_id_name(_array_element_id, _array_element_type)
    )


def get_array_element_name_from_type_site_id(array_element_type, site, array_element_id):
    """
    Get array element name from type, site and ID.

    Parameters
    ----------
    array_element_type: str
        Array element type.
    site: str
        Site name.
    array_element_id: str
        Array element ID.

    Returns
    -------
    str
        Array element name.
    """
    _short_site = validate_site_name(site)[0]
    _val_id = validate_array_element_id_name(array_element_id, array_element_type)
    return f"{array_element_type}{_short_site}-{_val_id}"


def get_array_element_type_from_name(name):
    """
    Get array element type from name, e.g. "LSTN", "MSTN".

    Parameters
    ----------
    name: str
        Array element name

    Returns
    -------
    str
        Array element type.
    """
    return _validate_name(name.split("-")[0], array_elements())


def get_design_model_from_name(name):
    """
    Get design model name from array element name.

    Note that this might not be correct and the preferred way is to use the
    model parameter 'design_model'.

    Parameters
    ----------
    name: str
       Array element name

    Returns
    -------
    str
        Design model name.
    """
    return f"{get_array_element_type_from_name(name)}-design"


def get_list_of_array_element_types(
    array_element_class="telescopes", site=None, observatory="CTAO"
):
    """
    Get list of array element types.

    Parameters
    ----------
    array_element_class: str
        Array element class
    site: str
        Site name (e.g., South or North).

    Returns
    -------
    list
        List of array element types.
    """
    return sorted(
        [
            key
            for key, value in array_elements().items()
            if value["collection"] == array_element_class
            and (site is None or value["site"] == site)
            and (observatory is None or value["observatory"] == observatory)
        ]
    )


def get_site_from_array_element_name(name):
    """
    Get site name from array element name.

    Parameters
    ----------
    name: str
        Array element name.

    Returns
    -------
    str, list
        Site name(s).
    """
    return array_elements()[get_array_element_type_from_name(name)]["site"]


def get_collection_name_from_array_element_name(name, array_elements_only=True):
    """
    Get collection name (e.g., telescopes, calibration_devices, sites) of array element from name.

    Parameters
    ----------
    name: str
        Array element name.
    array_elements_only: bool
        If True, only array elements are considered.

    Returns
    -------
    str
        Collection name .
    """
    try:
        return array_elements()[get_array_element_type_from_name(name)]["collection"]
    except ValueError:
        pass
    if name.startswith("OBS"):
        return "sites"
    try:
        validate_site_name(name)
        return "sites"
    except ValueError as exc:
        if array_elements_only:
            raise ValueError(f"Invalid array element name {name}") from exc
    if name in (
        "configuration_sim_telarray",
        "configuration_corsika",
        "Files",
        "Dummy-Telescope",
    ):
        return name

    raise ValueError(f"Invalid array element name {name}")


def get_simulation_software_name_from_parameter_name(
    par_name,
    simulation_software="sim_telarray",
):
    """
    Get the name used in the simulation software from the model parameter name.

    Name convention is expected to be defined in the schema.
    Returns the parameter name if no simulation software name is found.

    Parameters
    ----------
    par_name: str
        Model parameter name.
    simulation_software: str
        Simulation software name.

    Returns
    -------
    str
        Simtel parameter name.
    """
    _parameter_names = {**telescope_parameters(), **site_parameters()}

    try:
        _parameter = _parameter_names[par_name]
    except KeyError as err:
        raise KeyError(f"Parameter {par_name} without schema definition") from err

    try:
        for software in _parameter.get("simulation_software", []):
            if software.get("name") == simulation_software:
                return software.get("internal_parameter_name", par_name)
    except TypeError:  # catches cases for which 'simulation_software' is None
        pass
    return None


def get_parameter_name_from_simtel_name(simtel_name):
    """
    Get the model parameter name from the simtel parameter name.

    Assumes that both names are equal if not defined otherwise in names.py.

    Parameters
    ----------
    simtel_name: str
        Simtel parameter name.

    Returns
    -------
    str
        Model parameter name.
    """
    _parameters = {**telescope_parameters(), **site_parameters()}

    for par_name, par_info in _parameters.items():
        try:
            for software in par_info["simulation_software"]:
                if (
                    software["name"] == "sim_telarray"
                    and software["internal_parameter_name"] == simtel_name
                ):
                    return par_name
        except (KeyError, TypeError):  # catches cases for which 'simulation_software' is None
            pass
    return simtel_name


def simtel_config_file_name(
    site,
    model_version,
    array_name=None,
    telescope_model_name=None,
    label=None,
    extra_label=None,
):
    """
    sim_telarray config file name for a telescope.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam, ...
    model_version: str
        Version of the model.
    label: str
        Instance label.
    extra_label: str
        Extra label in case of multiple telescope config files.

    Returns
    -------
    str
        File name.
    """
    name = "CTA"
    name += f"-{array_name}" if array_name is not None else ""
    name += f"-{site}"
    name += f"-{telescope_model_name}" if telescope_model_name is not None else ""
    name += f"-{model_version}"
    name += f"_{label}" if label is not None else ""
    name += f"_{extra_label}" if extra_label is not None else ""
    name += ".cfg"
    return name


def simtel_single_mirror_list_file_name(
    site, telescope_model_name, model_version, mirror_number, label
):
    """
    sim_telarray mirror list file with a single mirror.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        North-LST-1, South-MST-FlashCam, ...
    model_version: str
        Version of the model.
    mirror_number: int
        Mirror number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = f"CTA-single-mirror-list-{site}-{telescope_model_name}-{model_version}"
    name += f"-mirror{mirror_number}"
    name += f"_{label}" if label is not None else ""
    name += ".dat"
    return name


def layout_telescope_list_file_name(name, label):
    """
    File name for files required at the RayTracing class.

    Parameters
    ----------
    name: str
        Name of the array.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    file_name = f"telescope_positions-{name}"
    file_name += f"_{label}" if label is not None else ""
    file_name += ".ecsv"
    return file_name


def generate_file_name(
    file_type,
    suffix,
    site,
    telescope_model_name,
    zenith_angle,
    azimuth_angle=None,
    off_axis_angle=None,
    source_distance=None,
    mirror_number=None,
    label=None,
    extra_label=None,
):
    """
    Generate a file name for output, config, or plotting.

    Used e.g., to generate camera-efficiency and ray-tracing output files.

    Parameters
    ----------
    file_type: str
        Type of file (e.g., config, output, plot)
    suffix: str
        File suffix
    site: str
        South or North.
    telescope_model_name: str
        LSTN-01, MSTS-01, ...
    zenith_angle: float
        Zenith angle (deg).
    azimuth_angle: float
        Azimuth angle (deg).
    off_axis_angle: float
        Off-axis angle (deg).
    source_distance: float
        Source distance (km).
    mirror_number: int
        Mirror number.
    label: str
        Instance label.
    extra_label: str
        Extra label.

    Returns
    -------
    str
        File name.
    """
    name = f"{file_type}-{site}-{telescope_model_name}"
    name += f"-d{source_distance:.1f}km" if source_distance is not None else ""
    name += f"-za{float(zenith_angle):.1f}deg"
    name += f"-off{off_axis_angle:.3f}deg" if off_axis_angle is not None else ""
    name += f"_azm{round(azimuth_angle):03}deg" if azimuth_angle is not None else ""
    name += f"_mirror{mirror_number}" if mirror_number is not None else ""
    name += f"_{label}" if label is not None else ""
    name += f"_{extra_label}" if extra_label is not None else ""
    name += f"{suffix}"
    return name


def sanitize_name(name):
    """
    Sanitize name to be a valid Python identifier.

    - Replaces spaces with underscores
    - Converts to lowercase
    - Removes characters that are not alphanumerics or underscores
    - If the name starts with a number, prepend an underscore

    Parameters
    ----------
    name: str
        name to be sanitized.

    Returns
    -------
    str:
        Sanitized name.

    Raises
    ------
    ValueError:
        if the string name can not be sanitized.
    """
    if name is None:
        # _logger.info("The string is None and can't be sanitized.")
        return name
    sanitized = name.lower()
    sanitized = sanitized.replace(" ", "_")
    # Remove characters that are not alphanumerics or underscores
    sanitized = re.sub(r"\W|^(?=\d)", "_", sanitized)
    if not sanitized.isidentifier():
        msg = f"The string {name} could not be sanitized."
        _logger.error(msg)
        raise ValueError(msg)
    return sanitized
