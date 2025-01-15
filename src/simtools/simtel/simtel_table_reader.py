#!/usr/bin/python3
"""Read tabular data in sim_telarray format and return as astropy table."""

import logging
import re

import astropy.units as u
from astropy.table import Table

from simtools.utils import general as gen

logger = logging.getLogger(__name__)


def _data_columns(parameter_name, n_columns, n_dim):
    """
    Get column information for a given parameter.

    Individual functions are adapted to the specific format of the sim_telarray tables.

    Parameters
    ----------
    parameter_name: str
        Model parameter name.
    n_columns: int
        Number of columns in the table.
    n_dim: list
        List of columns for n-dimensional tables defined by RPOL lines.

    Returns
    -------
    list, str
        List of columns for n-dimensional tables, description.
    """
    if parameter_name == "mirror_reflectivity":
        return _data_columns_mirror_reflectivity(n_columns, n_dim)
    if parameter_name in ("discriminator_pulse_shape", "fadc_pulse_shape"):
        return _data_columns_pulse_shape(n_columns)
    try:
        return globals()[f"_data_columns_{parameter_name}"]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported parameter for sim_telarray table reading: {parameter_name}"
        ) from exc


def _data_columns_atmospheric_profile():
    """Column representation for parameter atmospheric_profile."""
    return (
        [
            {"name": "altitude", "description": "Altitude", "unit": "km"},
            {"name": "density", "description": "Density", "unit": "g/cm^3"},
            {"name": "thickness", "description": "Thickness", "unit": "g/cm^2"},
            {
                "name": "refractive_index",
                "description": "Refractive index (n-1)",
                "unit": None,
            },
            {
                "name": "temperature",
                "description": "Temperature",
                "unit": "K",
            },
            {
                "name": "pressure",
                "description": "Pressure",
                "unit": "mbar",
            },
            {
                "name": "pw/w",
                "description": "Partial pressure of water vapor",
                "unit": None,
            },
        ],
        "Atmospheric profile",
    )


def _data_columns_pm_photoelectron_spectrum():
    """Column description for parameter pm_photoelectron_spectrum."""
    return (
        [
            {"name": "amplitude", "description": "Signal amplitude", "unit": None},
            {
                "name": "response",
                "description": "response without afterpulsing component",
                "unit": None,
            },
            {
                "name": "response_with_ap",
                "description": "response including afterpulsing component",
                "unit": None,
            },
        ],
        "Photoelectron spectrum",
    )


def _data_columns_quantum_efficiency():
    """Column description for parameter quantum_efficiency."""
    return (
        [
            {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
            {
                "name": "efficiency",
                "description": "Quantum efficiency",
                "unit": None,
            },
            {
                "name": "efficiency_rms",
                "description": "Quantum efficiency (standard deviation)",
                "unit": None,
            },
        ],
        "Quantum efficiency",
    )


def _data_columns_camera_filter():
    """Column description for parameter camera_filter."""
    return (
        [
            {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
            {
                "name": "transmission",
                "description": "Average transmission",
                "unit": None,
            },
        ],
        "Camera window transmission",
    )


def _data_columns_lightguide_efficiency_vs_wavelength():
    """Column description for parameter lightguide_efficiency_vs_wavelength."""
    return _data_columns_lightguide_efficiency_vs_incidence_angle()


def _data_columns_lightguide_efficiency_vs_incidence_angle():
    """Column description for (parameter lightguide_efficiency_vs_incidence_angle."""
    return (
        [
            {"name": "angle", "description": "Incidence angle", "unit": "deg"},
            {
                "name": "efficiency",
                "description": "Light guide efficiency",
                "unit": None,
            },
        ],
        "Light guide efficiency vs incidence angle",
    )


def _data_columns_mirror_reflectivity(n_columns, n_dim):
    """Column description for parameter mirror_reflectivity."""
    _columns = [
        {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
    ]
    if n_dim:
        for angle in n_dim:
            _columns.append(
                {
                    "name": f"reflectivity_{angle}deg",
                    "description": f"Mirror reflectivity at {angle} deg",
                    "unit": None,
                },
            )
    else:
        _columns.append(
            {
                "name": "reflectivity",
                "description": "Mirror reflectivity",
                "unit": None,
            },
        )
        if n_columns == 3:
            _columns.append(
                {
                    "name": "reflectivity_rms",
                    "description": "Mirror reflectivity (standard deviation)",
                    "unit": None,
                },
            )
        if n_columns == 4:
            _columns.append(
                {
                    "name": "reflectivity_min",
                    "description": "Mirror reflectivity (min)",
                    "unit": None,
                },
            )
            _columns.append(
                {
                    "name": "reflectivity_max",
                    "description": "Mirror reflectivity (max)",
                    "unit": None,
                },
            )

    return _columns, "Mirror reflectivity"


def _data_columns_pulse_shape(n_columns):
    """Column description for parameters discriminator_pulse_shape, fadc_pulse_shape."""
    _columns = [
        {"name": "time", "description": "Time", "unit": "ns"},
        {
            "name": "amplitude",
            "description": "Amplitude",
            "unit": None,
        },
    ]
    if n_columns == 3:
        _columns.append(
            {
                "name": "amplitude (low gain)",
                "description": "Amplitude (low gain)",
                "unit": None,
            },
        )

    return _columns, "Pulse shape"


def _data_columns_nsb_reference_spectrum():
    """Column description for parameter nsb_reference_spectrum."""
    return (
        [
            {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
            {
                "name": "differential photon rate",
                "description": "Differential photon rate",
                "unit": "1.e9 / (nm s m^2 sr)",
            },
        ],
        "NSB reference spectrum",
    )


def read_simtel_table(parameter_name, file_path):
    """
    Read sim_telarray table file for a given parameter.

    Parameters
    ----------
    parameter_name: str
        Model parameter name.
    file_path: Path
        Name (full path) of the sim_telarray table file.

    Returns
    -------
    Table
        Astropy table.
    """
    if parameter_name == "atmospheric_transmission":
        return _read_simtel_data_for_atmospheric_transmission(file_path)

    rows, meta_from_simtel, n_columns, n_dim = _read_simtel_data(file_path)
    columns_info, description = _data_columns(parameter_name, n_columns, n_dim)

    rows = _adjust_columns_length(rows, len(columns_info))

    metadata = {
        "Name": parameter_name,
        "File": str(file_path),
        "Description:": description,
        "Context_from_sim_telarray": meta_from_simtel,
    }

    table = Table(rows=rows, names=[col["name"] for col in columns_info])
    for col, info in zip(table.colnames, columns_info):
        table[col].unit = info.get("unit")
        table[col].description = info.get("description")
    table.meta.update(metadata)

    return table


def _adjust_columns_length(rows, n_columns):
    """
    Adjust row lengths to match the specified column count.

    - Truncate rows with extra columns beyond the specified count 'n_columns'.
    - Pad shorter rows with zeros.
    """
    return [row[:n_columns] + [0.0] * max(0, n_columns - len(row)) for row in rows]


def _read_simtel_data(file_path):
    """
    Read data, comments, and (if available) axis definition from sim_telarray table.

    Parameters
    ----------
    file_path: Path
        Path to the sim_telarray table file.

    Returns
    -------
    str, str, int, str
        data, metadata (comments), number of columns (max value), n-dimensional axis description.
    """
    logger.debug(f"Reading sim_telarray table from {file_path}")
    meta_lines = []
    data_lines = []
    n_dim_axis = None
    r_pol_axis = None

    lines = gen.read_file_encoded_in_utf_or_latin(file_path)

    for line in lines:
        stripped = line.strip()
        if "@RPOL@" in stripped:  # RPOL description for N-dimensional tables
            match = re.search(r"#@RPOL@\[(\w+)=\]", stripped)
            if match:
                r_pol_axis = match.group(1)
        elif r_pol_axis and r_pol_axis in stripped:  # N-dimensional axis description
            n_dim_axis = stripped.split("=")[1].split()
        elif stripped.startswith("#"):  # Metadata
            meta_lines.append(stripped.lstrip("#").strip())
        elif stripped:  # Data
            data_lines.append(stripped.split("%%%")[0].split("#")[0].strip())  # Remove comments

    rows = [[float(part) for part in line.split()] for line in data_lines]
    n_columns = max(len(row) for row in rows) if rows else 0

    return rows, "\n".join(meta_lines), n_columns, n_dim_axis


def _read_simtel_data_for_atmospheric_transmission(file_path):
    """
    Read data and comments from sim_telarray table for atmospheric_transmission.

    Parameters
    ----------
    file_path: Path
        Path to the sim_telarray table file.

    Returns
    -------
    astropy table
        Table with atmospheric transmission.
    """
    lines = lines = gen.read_file_encoded_in_utf_or_latin(file_path)

    observatory_level, height_bins = _read_header_line_for_atmospheric_transmission(
        lines, file_path
    )

    wavelengths = []
    heights = []
    extinctions = []
    meta_lines = []

    for line in lines:
        if line.startswith("#") or not line.strip():
            meta_lines.append(line.lstrip("#").strip())
            continue
        parts = line.split()
        try:
            wl = float(parts[0])
            for i, height in enumerate(height_bins):
                extinction_value = float(parts[i + 1])
                if extinction_value == 99999.0:
                    continue
                wavelengths.append(wl)
                heights.append(height)
                extinctions.append(extinction_value)
        except (ValueError, IndexError):
            logger.debug(f"Skipping malformed line: {line.strip()}")

    table = Table()
    table["wavelength"] = wavelengths
    table["altitude"] = heights
    table["extinction"] = extinctions

    table.meta.update(
        {
            "Name": "atmospheric_transmission",
            "File": str(file_path),
            "Description": "Atmospheric transmission",
            "Context_from_sim_telarray": "\n".join(meta_lines),
            "observatory_level": observatory_level,
        }
    )

    return table


def _read_header_line_for_atmospheric_transmission(lines, file_path):
    """Reader observatory level and height bins from header line for atmospheric transmission."""
    header_line = None
    observatory_level = None
    for line in lines:
        if "H2=" in line and "H1=" in line:
            match_h2 = re.search(r"H2=\s*([\d.]+)", line)
            if match_h2:
                observatory_level = float(match_h2.group(1)) * u.km

            if "H1=" in line:
                header_line = line.split("H1=")[-1].strip()
            break

    if header_line is None:
        raise ValueError(f"Header with 'H1=' not found file {file_path}")

    height_bins = [float(x.replace(",", "")) for x in header_line.split()]

    return observatory_level, height_bins
