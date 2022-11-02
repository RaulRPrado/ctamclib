import logging

__all__ = [
    "validate_model_version_name",
    "validate_simtel_mode_name",
    "validate_site_name",
    "validate_layout_array_name",
    "validate_telescope_model_name",
    "validate_camera_name",
    "convert_telescope_model_name_to_yaml",
    "split_telescope_model_name",
    "get_site_from_telescope_name",
    "ray_tracing_file_name",
    "simtel_telescope_config_file_name",
    "simtel_array_config_file_name",
    "simtel_single_mirror_list_file_name",
    "corsika_config_file_name",
    "corsika_output_file_name",
    "corsika_sub_log_file_name",
]


def validate_sub_system_name(name):
    """
    Validate a sub system name (optics structure or camera)

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validate_name(name, {**all_camera_names, **all_structure_names})


def validate_camera_name(name):
    """
    Validate a camera name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validate_name(name, all_camera_names)


def validate_telescope_id_name(name):
    """
    Validate a telescope ID name

    Valid names e.g.,
    - D
    - telescope ID

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """

    # FIXME: validate telescope id range
    if name == "D" or name.isdigit():
        return name

    _logger = logging.getLogger(__name__)
    msg = "Invalid telescope ID name {}".format(name)
    _logger.error(msg)
    raise ValueError(msg)


def validate_model_version_name(name):
    """
    Validate a model version name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validate_name(name, all_model_version_names)


def validate_simtel_mode_name(name):
    """
    Validate a sim_telarray mode name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validate_name(name, all_simtel_mode_names)


def validate_site_name(name):
    """
    Validate a site name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validate_name(name, all_site_names)


def validate_layout_array_name(name):
    """
    Validate a layout array name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    return validate_name(name, all_layout_array_names)


def validate_name(name, all_names):
    """
    Validate a name given the all_names options. For each key in all_names, a list of options is
    given. If name is in this list, the key name is returned.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str
    all_names: dict

    Returns
    -------
    str
        Validated name.
    """
    _logger = logging.getLogger(__name__)

    if not is_valid_name(name, all_names):
        msg = "Invalid name {}".format(name)
        _logger.error(msg)
        raise ValueError(msg)
    for main_name, list_of_names in all_names.items():
        if name.lower() in list_of_names + [main_name.lower()]:
            if name != main_name:
                _logger.debug("Correcting name {} -> {}".format(name, main_name))
            return main_name
    return None


def is_valid_name(name, all_names):
    """
    Parameters
    ----------
    name: str
    all_names: dict

    Returns
    -------
    bool
    """
    if not isinstance(name, str):
        return False
    for main_name in all_names.keys():
        if name.lower() in all_names[main_name] + [main_name.lower()]:
            return True
    return False


def validate_telescope_model_name(name):
    """
    Validate a telescope model name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    tel_class, tel_type = split_telescope_model_name(name)
    tel_class = validate_name(tel_class, all_telescope_class_names)
    if "flashcam" in tel_type:
        tel_type = tel_type.replace("flashcam", "FlashCam")
    if "nectarcam" in tel_type:
        tel_type = tel_type.replace("nectarcam", "NectarCam")
    if "1m" in tel_type:
        tel_type = tel_type.replace("1m", "1M")
    if "gct" in tel_type:
        tel_type = tel_type.replace("gct", "GCT")
    if "astri" in tel_type:
        tel_type = tel_type.replace("astri", "ASTRI")
    if "-d" in "-" + tel_type:
        tel_type = tel_type.replace("d", "D")

    return tel_class + "-" + tel_type


def split_telescope_model_name(name):
    """
    Split a telescope name into class and type.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str, str
       class (LST, MST, SST ...) and type (any complement).
    """
    name_parts = name.split("-")
    tel_class = name_parts[0]
    tel_type = "-".join(name_parts[1:])
    return tel_class, tel_type


def get_site_from_telescope_name(name):
    """
    Get site name (South or North) from the (validated) telescope name.

    Parameters
    ----------
    name: str
        Telescope name.

    Returns
    -------
    str
        Site name (South or North).
    """
    return validate_site_name(name.split("-")[0])


def validate_telescope_name_db(name):
    """
    Validate a telescope DB name.

    Raises
    ------
    ValueError
        If name is not valid.

    Parameters
    ----------
    name: str

    Returns
    -------
    str
        Validated name.
    """
    site = get_site_from_telescope_name(name)
    tel_model_name = "-".join(name.split("-")[1:])

    return f"{validate_site_name(site)}-{validate_telescope_model_name(tel_model_name)}"


def convert_telescope_model_name_to_yaml(name):
    """
    Get telescope name following the old convention (yaml files) from the current telescope name.

    Parameters
    ----------
    name: str
        Telescope model name.

    Returns
    -------
    str
        Telescope name (old convention).
    """
    tel_class, tel_type = split_telescope_model_name(name)
    new_name = tel_class + "-" + tel_type
    old_names = {
        "SST-D": "SST",
        "SST-1M": "SST-1M",
        "SST-ASTRI": "SST-2M-ASTRI",
        "SST-GCT": "SST-2M-GCT-S",
        "MST-FlashCam-D": "MST-FlashCam",
        "MST-NectarCam-D": "MST-NectarCam",
        "SCT-D": "SCT",
        "LST-D234": "LST",
        "LST-1": "LST",
    }

    if new_name not in old_names:
        raise ValueError("Telescope name {} could not be converted to yml names".format(name))
    else:
        return old_names[new_name]


all_telescope_class_names = {
    "SST": ["sst"],
    "MST": ["mst"],
    "SCT": ["sct"],
    "LST": ["lst"],
}

all_camera_names = {
    "SST": ["sst"],
    "ASTRI": ["astri"],
    "GCT": ["gct", "gct-s"],
    "1M": ["1m"],
    "FlashCam": ["flashcam", "flash-cam"],
    "NectarCam": ["nectarcam", "nectar-cam"],
    "SCT": ["sct"],
    "LST": ["lst"],
}

all_structure_names = {"Structure": ["Structure", "structure"]}

all_site_names = {"South": ["paranal", "south"], "North": ["lapalma", "north"]}

all_model_version_names = {
    "2015-07-21": [""],
    "2015-10-20-p1": [""],
    "prod4-v0.0": [""],
    "prod4-v0.1": [""],
    "2018-02-16": [""],
    "prod3_compatible": ["p3", "prod3", "prod3b"],
    "prod4": ["p4"],
    "post_prod3_updates": [""],
    "2016-12-20": [""],
    "2018-11-07": [""],
    "2019-02-22": [""],
    "2019-05-13": [""],
    "2019-11-20": [""],
    "2019-12-30": [""],
    "2020-02-26": [""],
    "2020-06-28": ["prod5"],
    "prod4-prototype": [""],
    "default": [],
    "Current": [],
    "Latest": [],
}

all_simtel_mode_names = {
    "RayTracing": ["raytracing", "ray-tracing"],
    "RayTracingSingleMirror": [
        "raytracing-singlemirror",
        "ray-tracing-singlemirror",
        "ray-tracing-single-mirror",
    ],
    "Trigger": ["trigger"],
}

all_layout_array_names = {
    "4LST": ["4-lst", "4lst"],
    "1LST": ["1-lst", "1lst"],
    "4MST": ["4-mst", "4mst"],
    "1MST": ["1-mst", "mst"],
    "4SST": ["4-sst", "4sst"],
    "1SST": ["1-sst", "sst"],
    "Prod5": ["prod5", "p5"],
    "TestLayout": ["test-layout"],
}


def simtools_instrument_name(site, telescope_class_name, sub_system_name, telescope_id_name):
    """
    Instrument name following gammasim-tools naming convention

    Parameters
    ----------
    site: str
        South or North.
    telescope_class_name: str
        LST, MST, ...
    sub_system_name: str
        FlashCam, NectarCam
    telescope_id_name: str
        telescope ID (e.g., D, numerial value)


    Returns
    -------
    instrumentname str
        instrument name

    """

    return (
        validate_site_name(site)
        + "-"
        + validate_name(telescope_class_name, all_telescope_class_names)
        + "-"
        + validate_sub_system_name(sub_system_name)
        + "-"
        + validate_telescope_id_name(telescope_id_name)
    )


def simtel_telescope_config_file_name(
    site, telescope_model_name, model_version, label, extra_label
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
    name = "CTA-{}-{}-{}".format(site, telescope_model_name, model_version)
    name += "_{}".format(label) if label is not None else ""
    name += "_{}".format(extra_label) if extra_label is not None else ""
    name += ".cfg"
    return name


def simtel_array_config_file_name(array_name, site, version, label):
    """
    sim_telarray config file name for an array.

    Parameters
    ----------
    array_name: str
        Prod5, ...
    site: str
        South or North.
    version: str
        Version of the model.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "CTA-{}-{}-{}".format(array_name, site, version)
    name += "_{}".format(label) if label is not None else ""
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
    name = "CTA-single-mirror-list-{}-{}-{}".format(site, telescope_model_name, model_version)
    name += "-mirror{}".format(mirror_number)
    name += "_{}".format(label) if label is not None else ""
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
    file_name = "telescope_positions-{}".format(name)
    file_name += "_{}".format(label) if label is not None else ""
    file_name += ".ecsv"
    return file_name


def ray_tracing_file_name(
    site,
    telescope_model_name,
    source_distance,
    zenith_angle,
    off_axis_angle,
    mirror_number,
    label,
    base,
):
    """
    File name for files required at the RayTracing class.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam, ...
    source_distance: float
        Source distance (km).
    zenith_angle: float
        Zenith angle (deg).
    off_axis_angle: float
        Off-axis angle (deg).
    mirror_number: int
        Mirror number. None if not single mirror case.
    label: str
        Instance label.
    base: str
        Photons, stars or log.

    Returns
    -------
    str
        File name.
    """
    name = "{}-{}-{}-d{:.1f}-za{:.1f}-off{:.3f}".format(
        base, site, telescope_model_name, source_distance, zenith_angle, off_axis_angle
    )
    name += "_mirror{}".format(mirror_number) if mirror_number is not None else ""
    name += "_{}".format(label) if label is not None else ""
    name += ".log" if base == "log" else ".lis"
    return name


def ray_tracing_results_file_name(site, telescope_model_name, source_distance, zenith_angle, label):
    """
    Ray tracing results file name.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam, ...
    source_distance: float
        Source distance (km).
    zenith_angle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "ray-tracing-{}-{}-d{:.1f}-za{:.1f}".format(
        site, telescope_model_name, source_distance, zenith_angle
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".ecsv"
    return name


def ray_tracing_plot_file_name(
    key, site, telescope_model_name, source_distance, zenith_angle, label
):
    """
    Ray tracing plot file name.

    Parameters
    ----------
    key: str
        Quantity to be plotted (d80_cm, d80_deg, eff_area or eff_flen)
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam, ...
    source_distance: float
        Source distance (km).
    zenith_angle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "ray-tracing-{}-{}-{}-d{:.1f}-za{:.1f}".format(
        site, telescope_model_name, key, source_distance, zenith_angle
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".pdf"
    return name


def camera_efficiency_results_file_name(site, telescope_model_name, zenith_angle, label):
    """
    Camera efficiency results file name.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam, ...
    zenith_angle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "camera-efficiency-{}-{}-za{:.1f}".format(site, telescope_model_name, zenith_angle)
    name += "_{}".format(label) if label is not None else ""
    name += ".ecsv"
    return name


def camera_efficiency_simtel_file_name(site, telescope_model_name, zenith_angle, label):
    """
    Camera efficiency simtel output file name.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam-D, ...
    zenith_angle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "camera-efficiency-{}-{}-za{:.1f}".format(site, telescope_model_name, zenith_angle)
    name += "_{}".format(label) if label is not None else ""
    name += ".dat"
    return name


def camera_efficiency_log_file_name(site, telescope_model_name, zenith_angle, label):
    """
    Camera efficiency log file name.

    Parameters
    ----------
    site: str
        South or North.
    telescope_model_name: str
        LST-1, MST-FlashCam-D, ...
    zenith_angle: float
        Zenith angle (deg).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "camera-efficiency-{}-{}-za{:.1f}".format(site, telescope_model_name, zenith_angle)
    name += "_{}".format(label) if label is not None else ""
    name += ".log"
    return name


def corsika_config_file_name(array_name, site, primary, zenith, view_cone, label=None):
    """
    CORSIKA config file name.

    Parameters
    ----------
    array_name: str
        Array name.
    site: str
        Paranal or LaPalma.
    primary: str
        Primary particle (e.g gamma, proton etc).
    zenith: float
        Zenith angle (deg).
    view_cone: list of float
        View cone limits (len = 2).
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    is_diffuse = view_cone[0] != 0 or view_cone[1] != 0

    name = "corsika-config_{}_{}_{}".format(site, array_name, primary)
    name += "_za{:d}-{:d}".format(int(zenith[0]), int(zenith[1]))
    name += "_cone{:d}-{:d}".format(int(view_cone[0]), int(view_cone[1])) if is_diffuse else ""
    name += "_{}".format(label) if label is not None else ""
    name += ".input"
    return name


def corsika_config_tmp_file_name(array_name, site, primary, zenith, view_cone, run, label=None):
    """
    CORSIKA config file name.

    Parameters
    ----------
    array_name: str
        Array name.
    site: str
        South or North.
    primary: str
        Primary particle (e.g gamma, proton etc).
    zenith: float
        Zenith angle (deg).
    view_cone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    is_diffuse = view_cone[0] != 0 or view_cone[1] != 0

    name = "corsika-config-run{}".format(run)
    name += "_{}_{}_{}".format(array_name, site, primary)
    name += "_za{:d}-{:d}".format(int(zenith[0]), int(zenith[1]))
    name += "_cone{:d}-{:d}".format(int(view_cone[0]), int(view_cone[1])) if is_diffuse else ""
    name += "_{}".format(label) if label is not None else ""
    name += ".txt"
    return name


def corsika_output_file_name(run, primary, array_name, site, zenith, azimuth, label=None):
    """
    CORSIKA output file name.

    Warnings
    --------
        zst extension is hardcoded here.

    Parameters
    ----------
    array_name: str
        Array name.
    site: str
        Paranal or LaPalma.
    zenith: float
        Zenith angle (deg).
    view_cone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "run{}_{}_za{:d}deg_azm{:d}deg-{}-{}".format(
        run, primary, int(zenith), int(azimuth), site, array_name
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".corsika.zst"
    return name


def corsika_output_generic_file_name(array_name, site, label=None):
    name = "run${RUNNR}_${PRMNAME}_za${ZA}deg_azm${AZM}deg"
    name += "-{}-{}".format(site, array_name)
    name += "_{}".format(label) if label is not None else ""
    name += ".corsika.zst"
    return name


def corsika_run_script_file_name(array_name, site, primary, run, label=None):
    """
    CORSIKA script file path.

    Parameters
    ----------
    array_name: str
        Array name.
    site: str
        Paranal or LaPalma.
    run: int
        RUn number.
    label: str
        Instance label.

    Returns
    -------
    str
        File path.
    """
    name = "run{}-corsika-{}-{}-{}".format(run, array_name, site, primary)
    name += "_{}".format(label) if label is not None else ""
    name += ".sh"
    return name


def corsika_run_log_file_name(array_name, site, primary, run, label=None):
    """
    CORSIKA script file name.

    Parameters
    ----------
    array_name: str
        Array name.
    site: str
        Paranal or LaPalma.
    primary: str
        Primary particle name.
    run: int
        RUn number.
    label: str
        Instance label.

    Returns
    -------
    str
        File path.
    """
    name = "log-corsika-run{}-{}-{}-{}".format(run, array_name, site, primary)
    name += "_{}".format(label) if label is not None else ""
    name += ".log"
    return name


def corsika_sub_log_file_name(array_name, site, primary, run, mode, label=None):
    """
    CORSIKA submission file name.

    Parameters
    ----------
    array_name: str
        Array name.
    site: str
        Paranal or LaPalma.
    primary: str
        Primary particle name.
    run: int
        RUn number.
    mode: str
        out or err.
    label: str
        Instance label.

    Returns
    -------
    str
        File path.
    """
    name = "log-sub-corsika-run{}-{}-{}-{}".format(run, array_name, site, primary)
    name += "_{}".format(label) if label is not None else ""
    if len(mode) > 0:
        name += "." + mode
    else:
        name += ".log"
    return name


def simtel_output_file_name(run, primary, array_name, site, zenith, azimuth, label=None):
    """
    sim_telarray output file name.

    Warning
    -------
        zst extension is hardcoded here.

    Parameters
    ----------
    array_name: str
        Array name.
    site: str
        Paranal or LaPalma.
    zenith: float
        Zenith angle (deg).
    view_cone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "run{}_{}_za{:d}deg_azm{:d}deg-{}-{}".format(
        run, primary, int(zenith), int(azimuth), site, array_name
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".simtel.zst"
    return name


def simtel_histogram_file_name(run, primary, array_name, site, zenith, azimuth, label=None):
    """
    sim_telarray histogram file name.

    Warning
    -------
        zst extension is hardcoded here.

    Parameters
    ----------
    array_name: str
        Array name.
    site: str
        Paranal or LaPalma.
    zenith: float
        Zenith angle (deg).
    view_cone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "run{}_{}_za{:d}deg_azm{:d}deg-{}-{}".format(
        run, primary, int(zenith), int(azimuth), site, array_name
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".hdata.zst"
    return name


def simtel_log_file_name(run, primary, array_name, site, zenith, azimuth, label=None):
    """
    sim_telarray histogram file name.

    Warning
    -------
        zst extension is hardcoded here.

    Parameters
    ----------
    array_name: str
        Array name.
    site: str
        Paranal or LaPalma.
    zenith: float
        Zenith angle (deg).
    view_cone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "run{}_{}_za{:d}deg_azm{:d}deg-{}-{}".format(
        run, primary, int(zenith), int(azimuth), site, array_name
    )
    name += "_{}".format(label) if label is not None else ""
    name += ".log"
    return name


def simtel_sub_log_file_name(run, primary, array_name, site, zenith, azimuth, mode, label=None):
    """
    sim_telarray submission log file name.

    Parameters
    ----------
    array_name: str
        Array name.
    site: str
        Paranal or LaPalma.
    zenith: float
        Zenith angle (deg).
    view_cone: list of float
        View cone limits (len = 2).
    run: int
        Run number.
    mode: str
        out or err
    label: str
        Instance label.

    Returns
    -------
    str
        File name.
    """
    name = "log-sub-run{}_{}_za{:d}deg_azm{:d}deg-{}-{}".format(
        run, primary, int(zenith), int(azimuth), site, array_name
    )
    name += "_{}".format(label) if label is not None else ""
    if len(mode) > 0:
        name += "." + mode
    else:
        name += ".log"
    return name
