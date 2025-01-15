"""Camera efficiency simulations and analysis."""

import logging
import re
from collections import defaultdict

import astropy.io.ascii
import numpy as np
from astropy.table import Table

from simtools.io_operations import io_handler
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simulator_camera_efficiency import SimulatorCameraEfficiency
from simtools.utils import names
from simtools.visualization import visualize

__all__ = ["CameraEfficiency"]


class CameraEfficiency:
    """
    Camera efficiency simulations and analysis.

    Parameters
    ----------
    simtel_path: str (or Path)
        Location of sim_telarray installation.
    db_config: dict
        Configuration for the database.
    label: str
        Instance label, optional.
    config_data: dict.
        Dict containing the configurable parameters.
    test: bool
        Is it a test instance (at the moment only affects the location of files).
    """

    def __init__(
        self,
        simtel_path,
        config_data,
        label,
        db_config,
        test=False,
    ):
        """Initialize the CameraEfficiency class."""
        self._logger = logging.getLogger(__name__)

        self._simtel_path = simtel_path
        self.label = label
        self.test = test

        self.io_handler = io_handler.IOHandler()
        self.telescope_model, self._site_model = self._initialize_simulation_models(
            config_data, db_config
        )
        self.output_dir = self.io_handler.get_output_directory(self.label, sub_dir="plots")

        self._results = None
        self._has_results = False

        self.config = self._configuration_from_args_dict(config_data)
        self._file = self._load_files()

    def __repr__(self):
        """Return string representation of the CameraEfficiency instance."""
        return f"CameraEfficiency(label={self.label})\n"

    def _initialize_simulation_models(self, config_data, db_config):
        """
        Initialize site and telescope models.

        Parameters
        ----------
        config_data: dict
            Dict containing the configurable parameters.

        Returns
        -------
        tuple
            Tuple containing the site and telescope models.
        """
        tel_model = TelescopeModel(
            site=config_data["site"],
            telescope_name=config_data["telescope"],
            mongo_db_config=db_config,
            model_version=config_data["model_version"],
            label=self.label,
        )
        site_model = SiteModel(
            site=config_data["site"],
            model_version=config_data["model_version"],
            mongo_db_config=db_config,
        )
        return tel_model, site_model

    def _configuration_from_args_dict(self, config_data):
        """
        Extract the configuration data from the args_dict.

        Zenith and azimuth angles are set to default values if not provided.

        Parameters
        ----------
        config_data: dict
            Dict containing the configurable parameters.

        Returns
        -------
        dict
            Configuration data.
        """
        zenith_angle = config_data.get("zenith_angle")
        if zenith_angle is not None:
            zenith_angle = zenith_angle.to("deg").value
        else:
            zenith_angle = 20.0
            self._logger.info(f"Setting zenith angle to default value {zenith_angle} deg")
        azimuth_angle = config_data.get("azimuth_angle")
        if azimuth_angle is not None:
            azimuth_angle = azimuth_angle.to("deg").value
        else:
            azimuth_angle = 0.0
            self._logger.info(f"Setting azimuth angle to default value {azimuth_angle} deg")

        return {
            "zenith_angle": zenith_angle,
            "azimuth_angle": azimuth_angle,
            "nsb_spectrum": config_data.get("nsb_spectrum", None),
        }

    def _load_files(self):
        """Define the variables for the file names, including the results, simtel and log file."""
        _file = {}
        for label, suffix in zip(
            ["results", "simtel", "log"],
            [".ecsv", ".dat", ".log"],
        ):
            file_name = names.generate_file_name(
                file_type=(
                    "camera-efficiency-table" if label == "results" else "camera-efficiency"
                ),
                suffix=suffix,
                site=self.telescope_model.site,
                telescope_model_name=self.telescope_model.name,
                zenith_angle=self.config["zenith_angle"],
                azimuth_angle=self.config["azimuth_angle"],
                label=self.label,
            )

            _file[label] = self.io_handler.get_output_directory(
                label=self.label,
                sub_dir="camera-efficiency",
            ).joinpath(file_name)
        return _file

    def simulate(self):
        """Simulate camera efficiency using testeff."""
        self._logger.info("Simulating CameraEfficiency")

        self.export_model_files()

        simtel = SimulatorCameraEfficiency(
            simtel_path=self._simtel_path,
            telescope_model=self.telescope_model,
            zenith_angle=self.config["zenith_angle"],
            file_simtel=self._file["simtel"],
            file_log=self._file["log"],
            label=self.label,
            nsb_spectrum=self.config["nsb_spectrum"],
            skip_correction_to_nsb_spectrum=self.config.get(
                "skip_correction_to_nsb_spectrum", False
            ),
        )
        simtel.run(test=self.test)

    def export_model_files(self):
        """Export model and config files to the output directory."""
        self.telescope_model.export_config_file()
        self.telescope_model.export_model_files()
        if not self.config.get("skip_correction_to_nsb_spectrum", False):
            self.telescope_model.export_nsb_spectrum_to_telescope_altitude_correction_file(
                model_directory=self.telescope_model.config_file_directory
            )

    def analyze(self, export=True, force=False):
        """
        Analyze camera efficiency output file and store the results in _results.

        Parameters
        ----------
        export: bool
            If True, results will be exported to a file automatically. Alternatively, export_results
            function can be used.
        force: bool
            If True, existing results files will be removed and analysis will be done again.
        """
        self._logger.info("Analyzing CameraEfficiency")

        if "results" in self._file and not force:
            self._logger.info("Results file exists and force=False - skipping analyze")
            self._read_results()
            return

        # List of parameters to be calculated and stored
        eff_pars = [
            "wl",
            "eff",
            "eff_atm",
            "qe",
            "ref",
            "masts",
            "filt",
            "pixel",
            "atm_trans",
            "cher",
            "nsb",
            "atm_corr",
            "nsb_site",
            "nsb_site_eff",
            "nsb_be",
            "nsb_be_eff",
            "C1",
            "C2",
            "C3",
            "C4",
            "C4x",
            "N1",
            "N2",
            "N3",
            "N4",
            "N4x",
        ]

        _results = defaultdict(list)

        # Search for at least 5 consecutive numbers to see that we are in the table
        re_table = re.compile("{0}{0}{0}{0}{0}".format(r"[-+]?[0-9]*\.?[0-9]+\s+"))
        with open(self._file["simtel"], encoding="utf-8") as file:
            for line in file:
                if re_table.match(line):
                    words = line.split()
                    numbers = [float(w) for w in words]
                    for i in range(len(eff_pars) - 10):
                        _results[eff_pars[i]].append(numbers[i])
                    C1 = numbers[8] * (400 / numbers[0]) ** 2  # noqa: N806
                    C2 = C1 * numbers[4] * numbers[5]  # noqa: N806
                    C3 = C2 * numbers[6] * numbers[7]  # noqa: N806
                    C4 = C3 * numbers[3]  # noqa: N806
                    c4x_value = C1 * numbers[3] * numbers[6] * numbers[7]
                    _results["C1"].append(C1)
                    _results["C2"].append(C2)
                    _results["C3"].append(C3)
                    _results["C4"].append(C4)
                    _results["C4x"].append(c4x_value)
                    N1 = numbers[14]  # noqa: N806
                    N2 = N1 * numbers[4] * numbers[5]  # noqa: N806
                    N3 = N2 * numbers[6] * numbers[7]  # noqa: N806
                    N4 = N3 * numbers[3]  # noqa: N806
                    n4x_value = N1 * numbers[3] * numbers[6] * numbers[7]
                    _results["N1"].append(N1)
                    _results["N2"].append(N2)
                    _results["N3"].append(N3)
                    _results["N4"].append(N4)
                    _results["N4x"].append(n4x_value)

        self._results = Table(_results)
        self._has_results = True

        print("\33[40;37;1m")
        self._logger.info(f"\n{self.results_summary()}")
        print("\033[0m")

        if export:
            self.export_results()

    def results_summary(self):
        """
        Print a summary of the results.

        Include a header for the zenith/azimuth settings and the NSB spectrum file which was used.
        The summary includes the various CTAO requirements and the final expected NSB pixel rate.
        """
        nsb_pixel_pe_per_ns, nsb_rate_ref_conditions = self.calc_nsb_rate()
        nsb_spectrum_text = (
            f"NSB spectrum file: {self.config['nsb_spectrum']}"
            if self.config["nsb_spectrum"]
            else "default sim_telarray spectrum."
        )
        return (
            f"Results summary for {self.telescope_model.name} at "
            f"zenith={self.config['zenith_angle']:.1f} deg, "
            f"azimuth={self.config['azimuth_angle']:.1f} deg\n"
            f"Using the {nsb_spectrum_text}\n"
            f"\nSpectrum weighted reflectivity: {self.calc_reflectivity():.4f}\n"
            "Camera nominal efficiency with gaps (B-TEL-1170): "
            f"{self.calc_camera_efficiency():.4f}\n"
            "Telescope total efficiency"
            f" with gaps (was A-PERF-2020): {self.calc_tel_efficiency():.4f}\n"
            "Telescope total Cherenkov light efficiency / sqrt(total NSB efficiency) "
            "(A-PERF-2025/B-TEL-0090): "
            f"{self.calc_tot_efficiency(self.calc_tel_efficiency()):.4f}\n"
            "Expected NSB pixel rate for the provided NSB spectrum: "
            f"{nsb_pixel_pe_per_ns:.4f} [p.e./ns]\n"
            "Expected NSB pixel rate for the reference NSB: "
            f"{nsb_rate_ref_conditions:.4f} [p.e./ns]\n"
        )

    def export_results(self):
        """Export results to a ecsv file."""
        if not self._has_results:
            self._logger.error("Cannot export results because they do not exist")
        else:
            self._logger.info(f"Exporting testeff table to {self._file['results']}")
            astropy.io.ascii.write(
                self._results, self._file["results"], format="basic", overwrite=True
            )
            _results_summary_file = (
                str(self._file["results"]).replace(".ecsv", ".txt").replace("-table-", "-summary-")
            )
            self._logger.info(f"Exporting summary results to {_results_summary_file}")
            with open(_results_summary_file, "w", encoding="utf-8") as file:
                file.write(self.results_summary())

    def _read_results(self):
        """Read existing results file and store it in _results."""
        table = astropy.io.ascii.read(self._file["results"], format="basic")
        self._results = table
        self._has_results = True

    def calc_tel_efficiency(self):
        """
        Calculate the telescope total efficiency including gaps (as defined in A-PERF-2020).

        Returns
        -------
        tel_efficiency: float
            Telescope efficiency
        """
        # Sum(C1) from 300 - 550 nm:
        c1_reduced_wl = self._results["C1"][[299 < wl_now < 551 for wl_now in self._results["wl"]]]
        c1_sum = np.sum(c1_reduced_wl)
        # Sum(C4) from 200 - 999 nm:
        c4_sum = np.sum(self._results["C4"])
        masts_factor = self._results["masts"][0]
        fill_factor = self.telescope_model.camera.get_camera_fill_factor()

        return fill_factor * (c4_sum / (masts_factor * c1_sum))

    def calc_camera_efficiency(self):
        """
        Calculate the camera nominal efficiency including gaps (as defined in B-TEL-1170).

        Returns
        -------
        cam_efficiency: float
            Wavelength-averaged camera efficiency
        """
        # Sum(C1) from 300 - 550 nm:
        c1_reduced_wl = self._results["C1"][[299 < wl_now < 551 for wl_now in self._results["wl"]]]
        c1_sum = np.sum(c1_reduced_wl)
        # Sum(C4x) from 300 - 550 nm:
        c4x_reduced_wl = self._results["C4x"][
            [299 < wl_now < 551 for wl_now in self._results["wl"]]
        ]
        c4x_sum = np.sum(c4x_reduced_wl)
        fill_factor = self.telescope_model.camera.get_camera_fill_factor()

        cam_efficiency_no_gaps = c4x_sum / c1_sum
        return cam_efficiency_no_gaps * fill_factor

    def calc_tot_efficiency(self, tel_efficiency):
        """
        Calculate the telescope total efficiency including gaps (as defined in A-PERF-2020).

        Parameters
        ----------
        tel_efficiency: float
            The telescope efficiency as calculated by calc_tel_efficiency()

        Returns
        -------
        Float
            Telescope total efficiency including gaps
        """
        # Sum(N1) from 300 - 550 nm:
        n1_reduced_wl = self._results["N1"][[299 < wl_now < 551 for wl_now in self._results["wl"]]]
        n1_sum = np.sum(n1_reduced_wl)
        # Sum(N4) from 200 - 999 nm:
        n4_sum = np.sum(self._results["N4"])
        masts_factor = self._results["masts"][0]
        fill_factor = self.telescope_model.camera.get_camera_fill_factor()

        tel_efficiency_nsb = fill_factor * (n4_sum / (masts_factor * n1_sum))

        return tel_efficiency / np.sqrt(tel_efficiency_nsb)

    def calc_reflectivity(self):
        """
        Calculate the Cherenkov spectrum weighted reflectivity in the range 300-550 nm.

        Returns
        -------
        Float
            Cherenkov spectrum weighted reflectivity (300-550 nm)
        """
        # Sum(C1) from 300 - 550 nm:
        c1_reduced_wl = self._results["C1"][[299 < wl_now < 551 for wl_now in self._results["wl"]]]
        c1_sum = np.sum(c1_reduced_wl)
        # Sum(C2) from 300 - 550 nm:
        c2_reduced_wl = self._results["C2"][[299 < wl_now < 551 for wl_now in self._results["wl"]]]
        c2_sum = np.sum(c2_reduced_wl)
        return c2_sum / c1_sum / self._results["masts"][0]

    def calc_nsb_rate(self):
        """
        Calculate the NSB rate.

        Returns
        -------
        nsb_rate_provided_spectrum: float
            NSB pixel rate in p.e./ns for the provided NSB spectrum
        nsb_rate_ref_conditions: float
            NSB pixel rate in p.e./ns for reference conditions
            (https://jama.cta-observatory.org/perspective.req#/items/26694?projectId=11)
        """
        nsb_rate_provided_spectrum = (
            np.sum(self._results["N4"])
            * self.telescope_model.camera.get_pixel_active_solid_angle()
            * self.telescope_model.get_on_axis_eff_optical_area().to("m2").value
            / self.telescope_model.get_parameter_value("telescope_transmission")[0]
        )

        # (integral is in ph./(m^2 ns sr) ) from 300 - 650 nm:
        n1_reduced_wl = self._results["N1"][[299 < wl_now < 651 for wl_now in self._results["wl"]]]
        n1_sum = np.sum(n1_reduced_wl)
        n1_integral_edges = self._results["N1"][
            [wl_now in [300, 650] for wl_now in self._results["wl"]]
        ]
        n1_integral_edges_sum = np.sum(n1_integral_edges)
        nsb_integral = 0.0001 * (n1_sum - 0.5 * n1_integral_edges_sum)
        nsb_rate_ref_conditions = (
            nsb_rate_provided_spectrum
            * self._site_model.get_parameter_value("nsb_reference_value")
            / nsb_integral
        )
        return nsb_rate_provided_spectrum, nsb_rate_ref_conditions

    def plot_efficiency(self, efficiency_type, save_fig=False):
        """
        Plot efficiency vs wavelength.

        Parameters
        ----------
        efficiency_type: str
            The type of efficiency to plot (Cherenkov 'C' or NSB 'N')
        save_fig: bool
            If True, the figure will be saved to a file.

        Returns
        -------
        fig
            The figure instance of pyplot
        """
        self._logger.info(f"Plotting {efficiency_type} efficiency vs wavelength")

        _col_type = "C" if efficiency_type == "Cherenkov" else "N"

        column_titles = {
            "wl": "Wavelength [nm]",
            f"{_col_type}1": rf"{_col_type}1: Cherenkov light on ground",
            f"{_col_type}2": rf"{_col_type}2: {_col_type}1 $\times$ ref. $\times$ masts",
            f"{_col_type}3": rf"{_col_type}3: {_col_type}2 $\times$ filter $\times$ lightguide",
            f"{_col_type}4": rf"{_col_type}4: {_col_type}3 $\times$ q.e.",
            f"{_col_type}4x": (
                rf"{_col_type}4x: {_col_type}1 $\times$ filter $\times$ lightguide $\times$ q.e."
            ),
        }

        table_to_plot = Table([self._results[col_now] for col_now in column_titles])

        for column_now, column_title in column_titles.items():
            table_to_plot.rename_column(column_now, column_title)

        y_title = f"{efficiency_type} light efficiency"
        if efficiency_type == "NSB":
            y_title = r"Diff. ph. rate [$10^{9} \times $ph/(nm s m$^2$ sr)]"
        plot = visualize.plot_table(
            table_to_plot,
            y_title=y_title,
            title=f"{self.telescope_model.name} response to {efficiency_type} light",
            no_markers=True,
        )
        if efficiency_type == "NSB":
            plot.gca().set_yscale("log")
            ylim = plot.gca().get_ylim()
            plot.gca().set_ylim(1e-3, ylim[1])
        if save_fig:
            self._save_plot(plot, efficiency_type.lower())
        return plot

    def _save_plot(self, fig, plot_title):
        """
        Save plot to pdf and png file.

        Parameters
        ----------
        fig
            The figure instance of pyplot
        plot_title: str
            The title of the plot
        """
        plot_file = self.output_dir.joinpath(
            self.label + "_" + self.telescope_model.name + "_" + plot_title
        )
        visualize.save_figure(fig, plot_file, log_title=f"{plot_title} efficiency")
