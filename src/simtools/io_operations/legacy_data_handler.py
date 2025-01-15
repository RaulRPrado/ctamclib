#!/usr/bin/python3
"""Reading of legacy data files (expect that this will be obsolete in future)."""

import logging

from astropy.table import Table

__all__ = [
    "read_legacy_data_as_table",
    "read_legacy_lst_single_pe",
]

logger = logging.getLogger(__name__)


def read_legacy_data_as_table(file_path, file_type):
    """
    Read legacy data file.

    Parameters
    ----------
    file_path: Path
        Path to the legacy data file.
    file_type: str
        Type of legacy data file.

    Returns
    -------
    Table
        Astropy table.

    Raises
    ------
    ValueError
        If unsupported legacy data file type.
    """
    logger.debug(f"Reading legacy data file of type {file_type} from {file_path}")

    try:
        return globals()[f"read_{file_type}"](file_path)
    except KeyError as exc:
        raise ValueError(f"Unsupported legacy data file type: {file_type}") from exc


def read_legacy_lst_single_pe(file_path):
    """
    Read LST single pe file (in legacy data format).

    File contains two columns: amplitude (in units of single p.e) and response.

    Parameters
    ----------
    file_path: Path
        Path to the legacy data file.

    Returns
    -------
    Table
        Astropy table.
    """
    return Table.read(file_path, format="ascii.csv", names=("amplitude", "response"))
