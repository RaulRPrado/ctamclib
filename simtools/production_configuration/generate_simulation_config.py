"""
Module defines the `SimulationConfig` class.

Used to configure and
generate simulation parameters for a specific grid point in a statistical error
evaluation setup. The class considers various parameters, such as azimuth,
elevation, and night sky background, to compute core scatter areas, viewcones,
and the required number of simulated events.

Key Components:
---------------
- `SimulationConfig`: Main class to handle simulation configuration for a grid point.
  - Attributes:
    - `grid_point` (dict): Contains azimuth, elevation, and night sky background.
    - `data_level` (str): The data level for the simulation (e.g., 'A', 'B', 'C').
    - `science_case` (str): The science case for the simulation.
    - `file_path` (str): Path to the FITS file used for statistical error evaluation.
    - `file_type` (str): Type of the FITS file ('On-source' or 'Offset').
    - `metrics` (dict, optional): Dictionary of metrics to evaluate.

"""

import os

import numpy as np

from simtools.production_configuration.calculate_statistical_errors_grid_point import (
    StatisticalErrorEvaluator,
)


class SimulationConfig:
    """
    Configures simulation parameters for a specific grid point.

    Parameters
    ----------
    grid_point : dict
        Dictionary representing a grid point with azimuth, elevation, and night sky background.
    data_level : str
        The data level (e.g., 'A', 'B', 'C') for the simulation configuration.
    science_case : str
        The science case for the simulation configuration.
    file_path : str
        Path to the FITS file for statistical error evaluation.
    file_type : str
        Type of the FITS file ('On-source' or 'Offset').
    metrics : dict, optional
        Dictionary of metrics to evaluate.
    """

    def __init__(
        self,
        grid_point: dict[str, float],
        data_level: str,
        science_case: str,
        file_path: str,
        file_type: str,
        metrics: dict[str, float] | None = None,
    ):
        """
        Initialize the simulation configuration for a grid point.

        Parameters
        ----------
        grid_point : dict
            A dictionary representing a grid point with azimuth,
              elevation, and night sky background.
        data_level : str
            The data level (e.g., 'A', 'B', 'C') for the simulation configuration.
        science_case : str
            The science case for the simulation configuration.
        file_path : str
            Path to the FITS file for statistical error evaluation.
        file_type : str
            Type of the FITS file ('On-source' or 'Offset').
        metrics : dict, optional
            Optional dictionary of metrics to evaluate.
        """
        self.grid_point = grid_point
        self.data_level = data_level
        self.science_case = science_case
        self.file_path = file_path
        self.file_type = file_type
        self.metrics = metrics or {}
        self.evaluator = StatisticalErrorEvaluator(file_path, file_type, metrics)
        self.simulation_params = {}

    def configure_simulation(self) -> dict[str, float]:
        """
        Configure the simulation parameters for the grid point, data level, and science case.

        Returns
        -------
        dict
            A dictionary with simulation parameters such as core scatter area,
              viewcone, and number of simulated events.
        """
        core_scatter_area = self._calculate_core_scatter_area()
        viewcone = self._calculate_viewcone()
        number_of_events = self.calculate_required_events()

        self.simulation_params = {
            "core_scatter_area": core_scatter_area,
            "viewcone": viewcone,
            "number_of_events": number_of_events,
        }

        return self.simulation_params

    def _calculate_core_scatter_area(self) -> float:
        """
        Calculate the core scatter area based on the grid point and data level.

        Returns
        -------
        float
            The core scatter area.
        """
        base_area = self._fetch_simulated_core_scatter_area()
        area_factor = self._get_area_factor_from_grid_point()
        return base_area * area_factor

    def _calculate_viewcone(self) -> float:
        """
        Calculate the viewcone based on the grid point conditions and data level.

        Returns
        -------
        float
            The viewcone value.
        """
        base_viewcone = self._fetch_simulated_viewcone()
        viewcone_factor = self._get_viewcone_factor_from_grid_point()
        return base_viewcone * viewcone_factor

    def _get_area_factor_from_grid_point(self) -> float:
        """
        Determine the area factor.

        The area factor is based on the grid point's azimuth,
          elevation, and night sky background.

        Returns
        -------
        float
            The area factor.
        """
        azimuth = self.grid_point.get("azimuth", 0)
        elevation = self.grid_point.get("elevation", 0)
        night_sky_background = self.grid_point.get("night_sky_background", 0)

        # TODO: implement
        return azimuth + elevation + night_sky_background

    def _get_viewcone_factor_from_grid_point(self) -> float:
        """
        Determine the viewcone factor.

        Determine the factor based on the grid point's azimuth,
          elevation, and night sky background.

        Returns
        -------
        float
            The viewcone factor.
        """
        azimuth = self.grid_point.get("azimuth", 0)
        elevation = self.grid_point.get("elevation", 0)
        night_sky_background = self.grid_point.get("night_sky_background", 0)

        # TODO: implement
        return azimuth + elevation + night_sky_background

    def _fetch_simulated_core_scatter_area(self) -> float:
        """
        Fetch the core scatter area from existing simulated files based on grid point conditions.

        Returns
        -------
        float
            The fetched core scatter area.
        """
        # Implement method
        return 1.0  # Placeholder value

    def _fetch_simulated_viewcone(self) -> float:
        """
        Fetch the viewcone from existing simulated files based on grid point conditions.

        Returns
        -------
        float
            The fetched viewcone.
        """
        # TODO: implement
        return 1

    def calculate_required_events(self) -> int:
        """
        Calculate the required number of simulated events based on statistical error metrics.

        Returns
        -------
        int
            The number of simulated events required.
        """
        # Obtain the statistical error evaluation metrics
        self.evaluator.calculate_metrics()
        metric_results = self.evaluator.metric_results

        # Calculate average uncertainty from metrics
        error_eff_area = metric_results.get("error_eff_area", {"uncertainties": [0.1]})
        avg_uncertainty = np.mean(error_eff_area["uncertainties"])

        # Calculate the base number of events from the evaluator
        base_events = self._fetch_existing_events()

        # Calculate required events
        uncertainty_factor = 1 / (1 - avg_uncertainty)
        if self.science_case == "science case 1":
            uncertainty_factor *= 1.5
        return int(base_events * uncertainty_factor)

    def _fetch_existing_events(self) -> int:
        """
        Fetch the number of existing simulated events from files based on grid point conditions.

        Returns
        -------
        int
            The number of existing simulated events.
        """
        return np.sum(self.evaluator.data.get("simulated_event_histogram", [0]))


if __name__ == "__main__":
    grid_point_config = {
        "azimuth": 45.0,
        "elevation": 30.0,
        "night_sky_background": 0.5,
    }

    BASE_PATH = "/Users/znb68/PD/CTA/"

    # Instantiate the StatisticalErrorEvaluator class a On-source file
    on_source_file = os.path.join(
        BASE_PATH, "gamma_onSource.N.BL-4LSTs15MSTs-MSTN_ID0.eff-0-CUT0.fits"
    )
    config = SimulationConfig(
        grid_point_config,
        "A",
        "high_precision",
        on_source_file,
        "On-source",
        metrics={
            "error_eff_area": 0.02,
            "error_sig_eff_gh": 0.02,
            "error_energy_estimate_bdt_reg_tree": 0.05,
            "error_gamma_ray_psf": 0.01,
            "error_image_template_methods": 0.03,
        },
    )
    params = config.configure_simulation()
    print("Simulation Parameters:", params)
