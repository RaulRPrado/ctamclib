#!/usr/bin/python3
"""Definition of site model."""

import logging
from pathlib import Path

from simtools.model.array_model import ArrayModel
from simtools.model.model_parameter import ModelParameter

__all__ = ["SiteModel"]


class SiteModel(ModelParameter):
    """
    SiteModel represents the MC model of an observatory site.

    Parameters
    ----------
    site: str
        Site name (e.g., South or North).
    mongo_db_config: dict
        MongoDB configuration.
    model_version: str
        Model version.
    label: str, optional
        Instance label. Important for output file naming.
    array_model : ArrayModel, optional
        Array model.
    """

    def __init__(
        self,
        site: str,
        mongo_db_config: dict,
        model_version: str,
        label: str | None = None,
        array_model: ArrayModel | None = None,
    ):
        """Initialize SiteModel."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SiteModel for site %s", site)
        ModelParameter.__init__(
            self,
            site=site,
            mongo_db_config=mongo_db_config,
            model_version=model_version,
            db=None,
            label=label,
        )
        self.array_model = array_model

    def get_reference_point(self):
        """
        Get reference point coordinates as dict.

        Returns
        -------
        dict
            Reference point coordinates as dict
        """
        return {
            "center_altitude": self.get_parameter_value_with_unit("reference_point_altitude"),
            "center_northing": self.get_parameter_value_with_unit("reference_point_utm_north"),
            "center_easting": self.get_parameter_value_with_unit("reference_point_utm_east"),
            "epsg_code": self.get_parameter_value("epsg_code"),
        }

    def get_corsika_site_parameters(self, config_file_style=False):
        """
        Get site-related CORSIKA parameters as dict.

        Parameters are returned with units wherever possible.

        Parameters
        ----------
        config_file_style bool
            Return using CORSIKA config file syntax

        Returns
        -------
        dict
            Site-related CORSIKA parameters as dict

        """
        if config_file_style:
            model_directory = Path("")
            if self.array_model:
                model_directory = self.array_model.get_config_directory()
            return {
                "OBSLEV": [
                    self.get_parameter_value_with_unit("corsika_observation_level").to_value("cm")
                ],
                # We always use a custom profile by filename, so this has to be set to 99
                "ATMOSPHERE": [99, "Y"],
                "IACT ATMOFILE": [
                    model_directory / self.get_parameter_value("atmospheric_profile")
                ],
                "MAGNET": [
                    self.get_parameter_value("geomag_horizontal"),
                    self.get_parameter_value("geomag_vertical"),
                ],
                "ARRANG": [self.get_parameter_value("geomag_rotation")],
            }

        return {
            "corsika_observation_level": self.get_parameter_value_with_unit(
                "corsika_observation_level"
            ),
            "geomag_horizontal": self.get_parameter_value_with_unit("geomag_horizontal"),
            "geomag_vertical": self.get_parameter_value_with_unit("geomag_vertical"),
            "geomag_rotation": self.get_parameter_value_with_unit("geomag_rotation"),
        }

    def get_array_elements_for_layout(self, layout_name):
        """
        Return list of array elements for a given array layout.

        Parameters
        ----------
        layout_name: str
            Name of the array layout

        Returns
        -------
        list
            List of array elements
        """
        layouts = self.get_parameter_value("array_layouts")
        for layout in layouts:
            if layout["name"] == layout_name.lower():
                return layout["elements"]
        raise ValueError(f"Array layout '{layout_name}' not found in '{self.site}' site model.")

    def get_list_of_array_layouts(self):
        """
        Get list of available array layouts.

        Returns
        -------
        list
            List of available array layouts
        """
        return [layout["name"] for layout in self.get_parameter_value("array_layouts")]

    def export_atmospheric_transmission_file(self, model_directory):
        """
        Export atmospheric transmission file.

        This method is needed because when CORSIKA is not piped to sim_telarray,
        the atmospheric transmission file is not written out to the model directory.
        This method allows to export it explicitly.

        Parameters
        ----------
        model_directory: Path
            Model directory to export the file to.
        """
        self.db.export_model_files(
            {
                "atmospheric_transmission_file": {
                    "value": self.get_parameter_value("atmospheric_profile"),
                    "file": True,
                }
            },
            model_directory,
        )
