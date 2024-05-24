import glob
import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml

_logger = logging.getLogger(__name__)

__all__ = [
    "get_site_from_telescope_name",
    "get_telescope_type_from_telescope_name",
    "generate_file_name",
    "layout_telescope_list_file_name",
    "sanitize_name",
    "simtel_single_mirror_list_file_name",
    "simtel_config_file_name",
    "validate_array_layout_name",
    "validate_site_name",
    "validate_telescope_id_name",
    "validate_telescope_name",
]


@lru_cache(maxsize=None)
def array_elements():
    """
    Load array elements from reference files and keep in cache.

    Returns
    -------
    dict
        Array elements.
    """
    base_path = Path(__file__).parent
    with open(base_path / "../schemas/array_elements.yml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["data"]


@lru_cache(maxsize=None)
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
    _sites = set(entry["site"] for entry in _array_elements.values())
    return {site: [site.lower()] for site in _sites}


array_layout_names = {
    "4LST": ["4-lst", "4lst"],
    "1LST": ["1-lst", "1lst"],
    "4MST": ["4-mst", "4mst"],
    "1MST": ["1-mst", "mst"],
    "4SST": ["4-sst", "4sst"],
    "1SST": ["1-sst", "sst"],
    "Prod5": ["prod5", "p5"],
    "TestLayout": ["test-layout"],
}


@lru_cache(maxsize=None)
def load_model_parameters(class_key_list):
    model_parameters = {}
    schema_files = glob.glob(str(Path(__file__).parent / "../schemas/model_parameters") + "/*.yml")
    for schema_file in schema_files:
        with open(schema_file, "r", encoding="utf-8") as f:
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
    return load_model_parameters(class_key_list=("Structure", "Camera"))


def validate_telescope_id_name(name):
    """
    Validate telescope ID. Allowed IDs are
    - design (for design telescopes or testing)
    - telescope ID (e.g., 1, 5, 15)
    - test (for testing)

    Parameters
    ----------
    name: str or int
        Telescope ID name.

    Returns
    -------
    str
        Validated telescope ID (added leading zeros, e.g., 1 is converted to 01).

    Raises
    ------
    ValueError
        If name is not valid.
    """

    if isinstance(name, int) or name.isdigit():
        return f"{int(name):02d}"
    if name.lower() in ("design", "test"):
        return str(name).lower()

    msg = f"Invalid telescope ID name {name}"
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


def validate_array_layout_name(name):
    """
    Validate array layout name.

    Parameters
    ----------
    name: str
        Layout array name.

    Returns
    -------
    str
        Validated name.
    """
    return _validate_name(name, array_layout_names)


def _validate_name(name, all_names):
    """
    Validate name given the all_names options. For each key in all_names, a list of options is \
    given. If name is in this list, the key name is returned.

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


def validate_telescope_name(name):
    """
    Validate telescope name (e.g., MSTN-design, MSTN-01).

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Validated name.
    """
    try:
        _tel_type, _tel_id = name.split("-")
    except ValueError as exc:
        msg = f"Invalid name {name}"
        raise ValueError(msg) from exc
    return _validate_name(_tel_type, array_elements()) + "-" + validate_telescope_id_name(_tel_id)


def get_telescope_name_from_type_site_id(telescope_type, site, telescope_id):
    """
    Get telescope name from type, site and ID.

    Parameters
    ----------
    telescope_type: str
        Telescope type.
    site: str
        Site name.
    telescope_id: str
        Telescope ID.

    Returns
    -------
    str
        Telescope name.
    """
    _short_site = validate_site_name(site)[0]
    _val_id = validate_telescope_id_name(telescope_id)
    return f"{telescope_type}{_short_site}-{_val_id}"


def get_telescope_type_from_telescope_name(name):
    """
    Get telescope type from name, e.g. "LSTN", "MSTN".

    Parameters
    ----------
    telescope_name: str
        Telescope name

    Returns
    -------
    str
        Telescope type.
    """
    return _validate_name(name.split("-")[0], array_elements())


def get_list_of_telescope_types(array_element_class="telescopes", site=None, observatory="CTAO"):
    """
    Get list of telescope types.

    Parameters
    ----------
    array_element_class: str
        Array element class
    site: str
        Site name (e.g., South or North).

    Returns
    -------
    list
        List of telescope types.
    """
    return [
        key
        for key, value in array_elements().items()
        if value["collection"] == array_element_class
        and (site is None or value["site"] == site)
        and (observatory is None or value["observatory"] == observatory)
    ]


def get_site_from_telescope_name(name):
    """
    Get site name from telescope name.

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Site name (South or North).
    """
    return array_elements()[get_telescope_type_from_telescope_name(name)]["site"]


def get_collection_name_from_array_element_name(name):
    """
    Get collection name(e.g., telescopes, calibration_devices) of array element from name

    Parameters
    ----------
    name: str
        Array element name.

    Returns
    -------
    str
        Collection name .
    """

    return array_elements()[get_telescope_type_from_telescope_name(name)]["collection"]


def get_simulation_software_name_from_parameter_name(
    par_name,
    simulation_software="sim_telarray",
    search_telescope_parameters=True,
    search_site_parameters=True,
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
    search_telescope_parameters: bool
        If True, telescope model parameters are included.
    search_site_parameters: bool
        If True, site model parameters are included.

    Returns
    -------
    str
        Simtel parameter name.
    """

    _parameter_names = {}
    if search_telescope_parameters:
        _parameter_names.update(telescope_parameters())
    if search_site_parameters:
        _parameter_names.update(site_parameters())

    try:
        _parameter = _parameter_names[par_name]
    except KeyError as err:
        _logger.error(f"Parameter {par_name} without schema definition")
        raise err

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

    # Convert to lowercase
    sanitized = name.lower()

    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")

    # Remove characters that are not alphanumerics or underscores
    sanitized = re.sub(r"\W|^(?=\d)", "_", sanitized)
    if not sanitized.isidentifier():
        msg = f"The string {name} could not be sanitized."
        _logger.error(msg)
        raise ValueError
    return sanitized
