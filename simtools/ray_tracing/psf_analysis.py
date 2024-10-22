"""
Module to analyse psf images (e.g. results from ray tracing simulations).

Main functionalities are: computing centroids, psf containers etc.

"""

import gzip
import logging
import shlex
import shutil
import subprocess
from math import fabs, pi, sqrt
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from simtools.utils.general import collect_kwargs, set_default_kwargs

__all__ = ["PSFImage"]


class PSFImage:
    """
    Image composed of list of photon positions (2D).

    Load photon list from sim_telarray file and compute centroids, psf containers, effective area,
    as well as plot the image as a 2D histogram.
    Internal units: photon positions in cm internally.

    Parameters
    ----------
    focal_length: float
        Focal length of the system in cm. If not given, PSF can only be computed in cm.
    total_scattered_area: float
        Scatter area of all photons in cm^2. If not given, effective area cannot be computed.
    containment_fraction: float
        Containment fraction for PSF calculation.
    simtel_path: str
        Path to sim_telarray installation.
    """

    __PSF_RADIUS = "Radius [cm]"
    __PSF_CUMULATIVE = "Cumulative PSF"

    def __init__(
        self,
        focal_length=None,
        total_scattered_area=None,
        containment_fraction=None,
        simtel_path=None,
    ):
        """Initialize PSFImage class."""
        self._logger = logging.getLogger(__name__)
        self.simtel_path = simtel_path

        self._total_photons = None
        self._number_of_detected_photons = None
        self._effective_area = None
        self.photon_pos_x = []
        self.photon_pos_y = []
        self.photon_r = []
        self.centroid_x = None
        self.centroid_y = None
        self._total_area = total_scattered_area
        self._stored_psf = {}
        try:
            self._cm_to_deg = 180.0 / pi / focal_length if focal_length is not None else None
        except ZeroDivisionError:
            self._cm_to_deg = None
            self._logger.warning("Focal length is zero; no conversion from cm to deg possible.")
        self._containment_fraction = containment_fraction

    def process_photon_list(self, photon_file, use_rx):
        """
        Read and process a photon list file generated by sim_telarray.

        Parameters
        ----------
        photons_file: str
            Name of sim_telarray file with photon list.
        use_rx: bool
            Use the RX method for analysis.
        """
        if use_rx:
            self._process_simtel_file_using_rx(photon_file)
        else:
            self.read_photon_list_from_simtel_file(photon_file)

    def _process_simtel_file_using_rx(self, photon_file):
        """
        Process a simtel file with photon lists using the RX method.

        Parameters
        ----------
        photons_file: str
            Name of sim_telarray file with photon list.
        """
        try:
            rx_output = subprocess.Popen(  # pylint: disable=consider-using-with
                shlex.split(
                    f"{self.simtel_path}/sim_telarray/bin/rx -f {self._containment_fraction:.2f} -v"
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            with gzip.open(photon_file, "rb") as _stdin:
                with rx_output.stdin:
                    shutil.copyfileobj(_stdin, rx_output.stdin)
                    try:
                        rx_output = rx_output.communicate()[0].splitlines()[-1:][0].split()
                    except IndexError as e:
                        raise IndexError(f"Unexpected output format from rx: {rx_output}") from e
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Photon list file not found: {photon_file}") from e

        try:
            self.set_psf(2 * float(rx_output[0]), fraction=self._containment_fraction, unit="cm")
            self.centroid_x = float(rx_output[1])
            self.centroid_y = float(rx_output[2])
            self._effective_area = float(rx_output[5])
        except IndexError as e:
            raise IndexError(f"Unexpected output format from rx: {rx_output}") from e
        except ValueError as e:
            raise ValueError(f"Invalid output format from rx: {rx_output}") from e

    def read_photon_list_from_simtel_file(self, photons_file):
        """
        Read photon list file generated by sim_telarray and store the photon positions (2D).

        Parameters
        ----------
        photons_file: str
            Name of sim_telarray file with photon list.

        Raises
        ------
        RuntimeError
            If photon positions X and Y are not compatible or are empty.

        """
        self._logger.info(f"Reading sim_telarray file {photons_file}")
        self._total_photons = 0
        if Path(photons_file).suffix == ".gz":
            file_open_function = gzip.open
        else:
            file_open_function = open
        with file_open_function(photons_file, "rb") as f:
            for line in f:
                self._process_simtel_line(line)

        if not self._is_photon_positions_ok():
            msg = "Problems reading Simtel file - invalid data"
            self._logger.error(msg)
            raise RuntimeError(msg)

        self.centroid_x = np.mean(self.photon_pos_x)
        self.centroid_y = np.mean(self.photon_pos_y)
        self._number_of_detected_photons = len(self.photon_pos_x)
        self._effective_area = (
            self._number_of_detected_photons * self._total_area / self._total_photons
        )
        self.photon_r = np.sort(
            np.sqrt(
                (self.photon_pos_x - self.centroid_x) ** 2
                + (self.photon_pos_y - self.centroid_y) ** 2
            )
        )

    def _is_photon_positions_ok(self):
        """
        Verify if the photon positions are ok.

        Returns
        -------
        bool
            True if photon positions are ok, False if they are not.
        """
        cond1 = len(self.photon_pos_x) != 0
        cond2 = len(self.photon_pos_y) != 0
        cond3 = len(self.photon_pos_x) == len(self.photon_pos_y)
        return cond1 and cond2 and cond3

    def _process_simtel_line(self, line):
        """
        Supporting function to read_photon_list_from_simtel_file.

        Parameters
        ----------
        line: str
            A line from the photon list file generated by sim_telarray.
        """
        words = line.split()
        if b"falling on an area of" in line:
            self._total_photons += int(words[4])
            total_area_in_file = float(words[14])
            if self._total_area is None:
                self._total_area = total_area_in_file
            elif total_area_in_file != self._total_area:
                self._logger.warning(
                    "Conflicting value of the total area found"
                    f" {self._total_area} != {total_area_in_file}"
                    " - Keeping the original value"
                )
        elif b"#" in line or len(words) == 0:
            # Skipping comments
            pass
        else:
            # Storing photon position from cols 2 and 3
            self.photon_pos_x.append(float(words[2]))
            self.photon_pos_y.append(float(words[3]))

    def get_effective_area(self, tel_transmission=1.0):
        """
        Return effective area pre calculated.

        Parameters
        ----------
        telescope_transmission : float
            Telescope transmission parameter.

        Returns
        -------
        float
            Pre-calculated effective area. None if it could not be calculated (e.g because the total
            scattering area was not set).
        """
        if "_effective_area" in self.__dict__ and self._effective_area is not None:
            return self._effective_area * tel_transmission

        self._logger.error("Effective Area could not be calculated")
        return None

    def set_effective_area(self, value):
        """
        Set effective area.

        Parameters
        ----------
        value: float
            Effective area

        """
        self._effective_area = value

    def get_psf(self, fraction=0.8, unit="cm"):
        """
        Return PSF.

        Parameters
        ----------
        fraction: float
            Fraction of photons within the containing radius.
        unit: str
            'cm' or 'deg'. 'deg' will not work if focal length was not set.

        Returns
        -------
        float:
            Containing diameter for a certain intensity fraction (PSF).

        """
        if unit == "deg" and self._cm_to_deg is None:
            self._logger.error("PSF cannot be computed in deg because focal length is not set")
            return None
        if fraction not in self._stored_psf:
            self._compute_psf(fraction)
        unit_factor = 1 if unit == "cm" else self._cm_to_deg
        return self._stored_psf[fraction] * unit_factor

    def set_psf(self, value, fraction=0.8, unit="cm"):
        """
        Set PSF calculated from other methods.

        Parameters
        ----------
        value: float
            PSF value to be set
        fraction: float
            Fraction of photons within the containing radius.
        unit: str
            'cm' or 'deg'. 'deg' will not work if focal length was not set.
        """
        if unit == "deg" and self._cm_to_deg is None:
            self._logger.error("PSF cannot be set in deg because focal length is not set")
            return
        unit_factor = 1 if unit == "cm" else 1.0 / self._cm_to_deg
        self._stored_psf[fraction] = value * unit_factor

    def _compute_psf(self, fraction):
        """
        Compute and store PSF.

        Parameters
        ----------
        fraction: float
            Fraction of photons within the containing radius
        """
        self._stored_psf[fraction] = self._find_psf(fraction)

    def _find_psf(self, fraction):
        """
        Try to find PSF by a smart algorithm first.

        If it fails, _find_radius_by_scanning is called and do it by brute force.

        Parameters
        ----------
        fraction: float
            Fraction of photons within the containing radius.

        Returns
        -------
        float:
            Diameter of the circular container with a certain fraction of the photons.

        """
        self._logger.debug(f"Finding PSF for fraction = {fraction}")

        x_pos_sq = [i**2 for i in self.photon_pos_x]
        y_pos_sq = [i**2 for i in self.photon_pos_y]
        x_pos_sig = sqrt(np.mean(x_pos_sq) - self.centroid_x**2)
        y_pos_sig = sqrt(np.mean(y_pos_sq) - self.centroid_y**2)
        radius_sig = sqrt(x_pos_sig**2 + y_pos_sig**2)

        target_number = fraction * self._number_of_detected_photons
        current_radius = 1.5 * radius_sig
        start_number = self._sum_photons_in_radius(current_radius)
        scale = 0.5 * sqrt(current_radius * current_radius / start_number)
        delta_number = start_number - target_number
        n_iter = 0
        max_iter = 1000
        tolerance = self._number_of_detected_photons / 1000.0
        found_radius = False
        while not found_radius and n_iter < max_iter:
            n_iter += 1
            dr = -delta_number * scale / sqrt(target_number)
            while current_radius + dr < 0:
                dr *= 0.5
            current_radius += dr
            current_number = self._sum_photons_in_radius(current_radius)
            delta_number = current_number - target_number
            found_radius = fabs(delta_number) < tolerance

        if found_radius:
            return 2 * current_radius

        self._logger.warning("Could not find PSF efficiently - trying by scanning")
        return 2 * self._find_radius_by_scanning(target_number, radius_sig)

    def _find_radius_by_scanning(self, target_number, radius_sig):
        """
        Find radius by scanning, aka brute force.

        Parameters
        ----------
        target_number: float
            Number of photons inside the diameter to be found.
        radius_sig: float
            Sigma of the radius to be used as scale.

        Returns
        -------
        float:
            Radius of the circle with target_number photons inside.
        """
        self._logger.debug("Finding PSF by scanning")

        def scan(dr, rad_min, rad_max):
            """
            Scan the image from rad_min to rad_max until it finds target_number photons inside.

            Scanning is done in steps of dr.

            Returns
            -------
            (float, float, float):
                Average radius, min radius, max radius of the interval where target_number photons
                are inside.

            Raises
            ------
            RuntimeError
                if radius is not found (found_radius is False)
            """
            r0, r1 = rad_min, rad_min + dr
            s0, s1 = 0, 0
            found_radius = False
            while not found_radius:
                s0, s1 = self._sum_photons_in_radius(r0), self._sum_photons_in_radius(r1)
                if s0 < target_number <= s1:
                    found_radius = True
                    break
                if r1 > rad_max:
                    break
                r0 += dr
                r1 += dr
            if found_radius:
                return (r0 + r1) / 2, r0, r1

            self._logger.error("Could not find PSF by scanning")
            raise RuntimeError

        # Run scan few times with smaller dr to optimize search.
        # Step 0
        radius, rad_min, rad_max = scan(0.1 * radius_sig, 0, 4 * radius_sig)
        # Step 1
        radius, rad_min, rad_max = scan(0.005 * radius_sig, rad_min, rad_max)
        return radius

    def _sum_photons_in_radius(self, radius):
        """Return the number of photons inside a certain radius."""
        return np.searchsorted(self.photon_r, radius)

    def get_image_data(self, centralized=True):
        """
        Provide image data (2D photon positions in cm) as lists.

        Parameters
        ----------
        centralized: bool
            Centroid of the image is set to (0, 0) if True.

        Returns
        -------
        (x, y), the photons positions in cm.
        """
        if centralized:
            x_pos_data = np.array(self.photon_pos_x) - self.centroid_x
            y_pos_data = np.array(self.photon_pos_y) - self.centroid_y
        else:
            x_pos_data = np.array(self.photon_pos_x)
            y_pos_data = np.array(self.photon_pos_y)
        d_type = {"names": ("X", "Y"), "formats": ("f8", "f8")}
        result = np.recarray((len(x_pos_data),), dtype=d_type)
        result.X = x_pos_data
        result.Y = y_pos_data
        return result

    def plot_image(self, centralized=True, file_name=None, **kwargs):
        """
        Plot 2D image as histogram (in cm).

        Parameters
        ----------
        centralized: bool
            Centroid of the image is set to (0, 0) if True.
        **kwargs:
            image_* for the histogram plot and psf_* for the psf circle.
        """
        data = self.get_image_data(centralized)

        kwargs = set_default_kwargs(
            kwargs,
            image_bins=80,
            image_cmap=plt.cm.gist_heat_r,
            psf_color="k",
            psf_fill=False,
            psf_lw=2,
            psf_ls="--",
        )
        kwargs_for_image = collect_kwargs("image", kwargs)
        kwargs_for_psf = collect_kwargs("psf", kwargs)

        ax = plt.gca()
        ax.set_xlabel("X Position (cm)")
        ax.set_ylabel("Y Position (cm)")
        ax.hist2d(data["X"], data["Y"], **kwargs_for_image)
        ax.set_aspect("equal", "datalim")

        # PSF circle (80%)
        center = (0, 0) if centralized else (self.centroid_x, self.centroid_y)
        circle = plt.Circle(center, self.get_psf(0.8) / 2, **kwargs_for_psf)
        ax.add_artist(circle)

        ax.axhline(0, color="k", linestyle="--", zorder=3, linewidth=0.5)
        ax.axvline(0, color="k", linestyle="--", zorder=3, linewidth=0.5)

        if file_name is not None:
            plt.savefig(file_name)
            plt.close()

    def get_cumulative_data(self, radius=None):
        """
        Provide cumulative data (intensity vs radius).

        Parameters
        ----------
        radius: array
            Array with radius calculate the cumulative PSF in distance units.

        Returns
        -------
        (radius, intensity)
        """
        if radius is not None:
            radius_all = radius.to(u.cm).value if isinstance(radius, u.Quantity) else radius
        else:
            radius_all = list(np.linspace(0, 1.6 * self.get_psf(0.8), 30))

        intensity = []
        for rad in radius_all:
            intensity.append(self._sum_photons_in_radius(rad) / self._number_of_detected_photons)
        d_type = {
            "names": (self.__PSF_RADIUS, self.__PSF_CUMULATIVE),
            "formats": ("f8", "f8"),
        }
        result = np.recarray((len(radius_all),), dtype=d_type)
        result[self.__PSF_RADIUS] = radius_all
        result[self.__PSF_CUMULATIVE] = intensity

        return result

    def plot_cumulative(self, file_name=None, d80=None, **kwargs):
        """Plot cumulative data (intensity vs radius).

        Parameters
        ----------
        **kwargs:
            image_* for the histogram plot and psf_* for the psf circle.
        """
        data = self.get_cumulative_data()
        ax = plt.gca()
        plt.tight_layout(pad=1.5)
        ax.set_xlabel("Radius (cm)")
        ax.set_ylabel("Contained light %")
        plt.plot(data[self.__PSF_RADIUS], data[self.__PSF_CUMULATIVE], **kwargs)
        plt.axvline(x=self.get_psf(0.8) / 2, color="b", linestyle="--", linewidth=1)
        if d80 is not None:
            plt.axvline(x=d80 / 2.0, color="r", linestyle="--", linewidth=1)
        if file_name is not None:
            plt.savefig(file_name)
            plt.close()
