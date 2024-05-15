#!/usr/bin/python3

import math

from simtools.utils import names

__all__ = [
    "compute_telescope_transmission",
    "is_two_mirror_telescope",
]


def compute_telescope_transmission(pars, off_axis):
    """
    Compute telescope transmission (0 < T < 1) for a given set of parameters as defined by \
    the MC model and for a given off-axis angle.

    Parameters
    ----------
    pars: list of float
        Parameters of the telescope transmission. Len(pars) should be 4.
    off_axis: float
        Off-axis angle in deg.

    Returns
    -------
    float
        Telescope transmission.
    """
    _deg_to_rad = math.pi / 180.0
    if pars[1] == 0:
        return pars[0]

    t = math.sin(off_axis * _deg_to_rad) / (pars[3] * _deg_to_rad)
    return pars[0] / (1.0 + pars[2] * t ** pars[4])


def is_two_mirror_telescope(telescope_model_name):
    """
    Check if the telescope is a two mirror design.

    Parameters
    ----------
    telescope_model_name: str
        Telescope model name (ex. LSTN-01).

    Returns
    -------
    bool
        True if the telescope is a two mirror one.
    """
    tel_type = names.get_telescope_type_from_telescope_name(telescope_model_name)
    if "SST" in tel_type or "SCT" in tel_type:
        return True
    return False
