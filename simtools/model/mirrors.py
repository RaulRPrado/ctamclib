import logging

import astropy.units as u
from astropy.table import Table

__all__ = ["InvalidMirrorListFile", "Mirrors"]


class InvalidMirrorListFile(Exception):
    """Exception for invalid mirror list file."""


class Mirrors:
    """
    Mirrors class, created from a mirror list file.

    Parameters
    ----------
    mirror_list_file: str
        mirror list in sim_telarray or ecsv format (with panel focal length only).
    """

    def __init__(self, mirror_list_file):
        """
        Initialize Mirrors.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Mirrors Init")

        self._mirror_table = Table()
        self.diameter = None
        self.shape = None
        self.number_of_mirrors = 0

        self._mirror_list_file = mirror_list_file
        self._read_mirror_list()

    def _read_mirror_list(self):
        """
        Read the mirror lists from disk and store the data. Allow reading of mirror lists in \
        sim_telarray and ecsv format
        """

        if str(self._mirror_list_file).find("ecsv") > 0:
            self._read_mirror_list_from_ecsv()
        else:
            self._read_mirror_list_from_sim_telarray()

    def _read_mirror_list_from_ecsv(self):
        """
        Read the mirror list in ecsv format and store the data.

        Raises
        ------
        InvalidMirrorListFile
            If number of mirrors is 0.
        """
        # Getting mirror parameters from mirror list.

        self._mirror_table = Table.read(self._mirror_list_file, format="ascii.ecsv")
        self._logger.debug(f"Reading mirror properties from {self._mirror_list_file}")

        self.shape = u.Quantity(self._mirror_table["shape_code"])[0].value
        self.diameter = u.Quantity(self._mirror_table["diameter"])[0].value
        self.number_of_mirrors = len(self._mirror_table["focal_length"])

        self._logger.debug(f"Shape = {self.shape}")
        self._logger.debug(f"Diameter = {self.diameter}")
        self._logger.debug(f"Number of Mirrors = {self.number_of_mirrors}")

        if self.number_of_mirrors == 0:
            msg = "Problem reading mirror list file"
            self._logger.error(msg)
            raise InvalidMirrorListFile()

    def _read_mirror_list_from_sim_telarray(self):
        """
        Read the mirror list in sim_telarray format and store the data.

        Raises
        ------
        InvalidMirrorListFile
            If number of mirrors is 0.
        """

        self._mirror_table = Table.read(self._mirror_list_file, 
                            format="ascii.no_header", 
                            names =["pos_x","pos_y", "diameter","focal_length", "shape","pos_z","sep","mirror_number"])
        self.shape = self._mirror_table["shape"][0]
        self.diameter = self._mirror_table["diameter"][0]
        self.number_of_mirrors = len(self._mirror_table["focal_length"])

        if self.number_of_mirrors == 0:
            msg = "Problem reading mirror list file"
            self._logger.error(msg)
            raise InvalidMirrorListFile()

    def get_single_mirror_parameters(self, number):
        """
        Get parameters for a single mirror given by number.

        Parameters
        ----------
        number: int
            Mirror number of desired parameters.

        Returns
        -------
        (pos_x, pos_y, diameter, focal_length, shape): tuple of float
            X, Y positions, diameter, focal length and shape.
        """

        if number > self.number_of_mirrors - 1:
            self._logger.error("Mirror number is out range")
            return None
        return (
            self._mirror_table["pos_x"][number],
            self._mirror_table["pos_y"][number],
            self._mirror_table["diameter"][number],
            self._mirror_table["focal_length"][number],
            self._mirror_table["shape"][number],
        )

    def plot_mirror_layout(self):
        """
        Plot the mirror layout.

        TODO
        """
