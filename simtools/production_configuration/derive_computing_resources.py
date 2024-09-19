"""
Module for estimating compute and storage resources required for simulations.

The estimation can be based on historical data from existing simulations or
by making guesses per event based on the provided number of events.

Attributes
----------
grid_point : dict
    Dictionary containing parameters such as azimuth, elevation, and night sky background.
simulation_params : dict
    Dictionary containing simulation parameters, including the number of events.
existing_data : list of dict, optional
    List of dictionaries with historical data of compute and storage resources
      for existing simulations.

Methods
-------
estimate_resources() -> dict:
    Estimates resources required for the simulation using the number of events
      from simulation_params.
interpolate_resources() -> dict:
    Interpolates resources based on existing data.
guess_resources_per_event() -> dict:
    Estimates resources per event using grid point parameters.

Example Usage
-------------

    grid_point_config = {
        "azimuth": 60.0,
        "elevation": 45.0,
        "night_sky_background": 0.3,
    }

    simulation_params = {
        "number_of_events": 1e9
    }

    existing_data = [
        {
            "azimuth": 60.0,
            "elevation": 45.0,
            "nsb": 0.3,
            "compute_hours": 5000.0,
            "storage_gb": 500.0,
            "events": 1e9,
        },
    ]

    estimator = ResourceEstimator(grid_point=grid_point_config,
      simulation_params=simulation_params, existing_data=existing_data)
    resources = estimator.estimate_resources()
    print("Estimated Resources:", resources)
"""

import numpy as np
import yaml

from simtools.utils import names


class ResourceEstimator:
    """
    Estimates compute and storage resources required for simulations.

    The estimation can be based on historical data from existing simulations or
    by making guesses per event using the number of events provided.

    Attributes
    ----------
    grid_point : dict
        Dictionary containing parameters such as azimuth, elevation, and night sky background.
    simulation_params : dict
        Dictionary containing simulation parameters, including the number of events and site.
    existing_data : list of dict, optional
        List of dictionaries with historical data of compute and storage
          resources for existing simulations.

    Methods
    -------
    estimate_resources() -> dict:
        Estimates resources required for the simulation using the number of
        events from simulation_params.
    interpolate_resources() -> dict:
        Interpolates resources based on existing data.
    guess_resources_per_event() -> dict:
        Estimates resources per event using grid point parameters.
    """

    def __init__(
        self,
        grid_point: dict[str, float],
        simulation_params: dict[str, float],
        existing_data: list[dict] | None = None,
        lookup_file: str = "resource_estimates.yaml",
    ):
        """
        Initialize the resource estimator.

        Initialize with grid point parameters,
        simulation parameters, and optional existing data.

        Parameters
        ----------
        grid_point : dict
            Dictionary containing grid point parameters such as azimuth,
              elevation, and night sky background.
        simulation_params : dict
            Dictionary containing simulation parameters, including the number of events and site.
        existing_data : list of dict, optional
            List of dictionaries with historical data of compute and storage
              resources for existing simulations.
        """
        self.grid_point = grid_point
        self.simulation_params = simulation_params
        self.existing_data = existing_data or []
        self.lookup_table = self.load_lookup_table(lookup_file)

        self.site = names.validate_site_name(self.simulation_params["site"])

    @staticmethod
    def load_lookup_table(lookup_file: str) -> dict:
        """
        Load the lookup table from a YAML file.

        Parameters
        ----------
        lookup_file : str
            Path to the YAML file containing the lookup table.

        Returns
        -------
        dict
            Dictionary containing the lookup table.
        """
        with open(lookup_file, encoding="utf-8") as file:
            return yaml.safe_load(file)

    def estimate_resources(self) -> dict:
        """
        Estimate the compute and storage resources required for the simulation.

        Returns
        -------
        dict
            A dictionary with estimates for compute and storage resources.
        """
        number_of_events = self.simulation_params.get("number_of_events", 0)
        if self.existing_data:
            return self.interpolate_resources(number_of_events)

        return self.guess_resources_per_event(number_of_events)

    def interpolate_resources(self, number_of_events: int) -> dict:
        """
        Interpolate resources required for the simulation from existing data.

        Parameters
        ----------
        number_of_events : int
            The number of events for which to interpolate resources.

        Returns
        -------
        dict
            A dictionary with interpolated estimates for compute and storage resources.
        """
        azimuth = self.grid_point["azimuth"]
        elevation = self.grid_point["elevation"]
        nsb = self.grid_point["night_sky_background"]

        closest_data = min(
            self.existing_data,
            key=lambda x: (
                abs(x["azimuth"] - azimuth) + abs(x["elevation"] - elevation) + abs(x["nsb"] - nsb)
            ),
        )

        compute_hours = closest_data["compute_hours"] * (number_of_events / closest_data["events"])
        storage_gb = closest_data["storage_gb"] * (number_of_events / closest_data["events"])

        return {"compute_hours": compute_hours, "storage_gb": storage_gb}

    def guess_resources_per_event(self, number_of_events: int) -> dict:
        """
        Estimate resources for the simulation based on grid point parameters and per-event guess.

        Parameters
        ----------
        number_of_events : int
            The number of events for which to estimate resources.

        Returns
        -------
        dict
            A dictionary with guessed estimates for compute and storage resources.
        """
        elevation = self.grid_point["elevation"]
        elevations = sorted(self.lookup_table[self.site].keys())

        if elevation <= elevations[0]:
            compute_hours_per_event = self.lookup_table[self.site][elevations[0]][
                "compute_hours_per_event"
            ]
            storage_gb_per_event = self.lookup_table[self.site][elevations[0]][
                "storage_gb_per_event"
            ]
        elif elevation >= elevations[-1]:
            compute_hours_per_event = self.lookup_table[self.site][elevations[-1]][
                "compute_hours_per_event"
            ]
            storage_gb_per_event = self.lookup_table[self.site][elevations[-1]][
                "storage_gb_per_event"
            ]
        else:
            lower_bound = max(e for e in elevations if e <= elevation)
            upper_bound = min(e for e in elevations if e >= elevation)
            lower_values = self.lookup_table[self.site][lower_bound]
            upper_values = self.lookup_table[self.site][upper_bound]

            compute_hours_per_event = np.interp(
                elevation,
                [lower_bound, upper_bound],
                [lower_values["compute_hours_per_event"], upper_values["compute_hours_per_event"]],
            )
            storage_gb_per_event = np.interp(
                elevation,
                [lower_bound, upper_bound],
                [lower_values["storage_gb_per_event"], upper_values["storage_gb_per_event"]],
            )

        compute_hours = number_of_events * compute_hours_per_event
        storage_gb = number_of_events * storage_gb_per_event

        return {"compute_hours": compute_hours, "storage_gb": storage_gb}