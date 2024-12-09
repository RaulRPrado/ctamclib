#!/usr/bin/python3

r"""
Generate a grid of simulation points using flexible axes definitions.

This application generates a grid of simulation points based on input parameters
such as energy, azimuth, zenith angle, night-sky background, and camera offset.
It can also convert the generated points to RA/Dec coordinates if the selected
coordinate system is 'ra_dec'.

Command line arguments
----------------------
axes (str, required)
    Path to a YAML or JSON file defining the axes of the grid.
ctao_data_level (str, required)
    The data level for the grid generation (e.g., 'A', 'B', 'C').
science_case (str, required)
    The science case for the grid generation (e.g., 'high_precision').
coordinate_system (str, optional, default='zenith_azimuth')
    The coordinate system for the grid generation ('zenith_azimuth' or 'ra_dec').
latitude (float, required)
    Latitude of the observing location in degrees.
longitude (float, required)
    Longitude of the observing location in degrees.
height (float, required)
    Height of the observing location in meters.
observing_time (str, optional, default=now)
    Time of the observation (format: 'YYYY-MM-DD HH:MM:SS').

Example
-------
To generate a grid of simulation points, execute the script as follows:

.. code-block:: console

    simtools-production-generate-grid --site North --model_version "6.0.0"\
      --axes  tests/resources/production_grid_generation_axes_definition.yaml\
      --ctao_data_level "B" \
      --science_case "high_precision" \
      --coordinate_system "ra_dec" --observing_time "2017-09-16 00:00:00"

The output will display the generated grid points and their RA/Dec coordinates
(if applicable).
"""

import json
import logging
from pathlib import Path

import yaml
from astropy.coordinates import EarthLocation
from astropy.time import Time

import simtools.production_configuration.generate_production_grid as gridgen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.model.site_model import SiteModel


def _parse(label, description):
    """
    Parse command line configuration.

    Parameters
    ----------
    label : str
        Label describing the application.
    description : str
        Description of the application.

    Returns
    -------
    CommandLineParser
        Command line parser object.
    """
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--axes",
        type=str,
        required=True,
        help="Path to a YAML or JSON file defining the grid axes.",
    )
    config.parser.add_argument(
        "--ctao_data_level", type=str, required=True, help="Data level (e.g., 'A', 'B', 'C')."
    )
    config.parser.add_argument(
        "--science_case", type=str, required=True, help="Science case for the grid generation."
    )
    config.parser.add_argument(
        "--coordinate_system",
        type=str,
        default="zenith_azimuth",
        help="Coordinate system ('zenith_azimuth' or 'ra_dec').",
    )
    config.parser.add_argument(
        "--observing_time",
        type=str,
        required=False,
        help="Time of the observation (format: 'YYYY-MM-DD HH:MM:SS').",
    )
    config.parser.add_argument(
        "--output_file",
        type=str,
        default="grid_output.json",
        help="Output file for the generated grid points (default: 'grid_output.json').",
    )

    return config.initialize(db_config=True, simulation_model=["version", "site"])


def load_axes(file_path: str):
    """
    Load axes definitions from a YAML or JSON file.

    Parameters
    ----------
    file_path : str
        Path to the axes YAML or JSON file.

    Returns
    -------
    list[dict]
        List of axes definitions with Quantity values.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Axes file {file_path} not found.")

    with open(file_path, encoding="utf-8") as file:
        if file_path.endswith((".yaml", ".yml")):
            axes = yaml.safe_load(file)
        elif file_path.endswith(".json"):
            axes = json.load(file)
        else:
            raise ValueError("Unsupported file format. Use a YAML or JSON file.")

    return axes


def main():
    """Run the Grid Generation application."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label,
        "Generate a grid of simulation points using flexible axes definitions.",
    )

    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)

    output_path = io_handler.IOHandler().get_output_directory(label)
    output_filepath = Path(output_path).joinpath(f"{args_dict['output_file']}")

    axes = load_axes(args_dict["axes"])
    site_model = SiteModel(
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        site=args_dict["site"],
    )

    ref_lat = site_model.get_parameter_value_with_unit("reference_point_latitude")
    ref_long = site_model.get_parameter_value_with_unit("reference_point_longitude")
    altitude = site_model.get_parameter_value_with_unit("reference_point_altitude")

    observing_location = EarthLocation(lat=ref_lat, lon=ref_long, height=altitude)

    observing_time = (
        Time(args_dict["observing_time"]) if args_dict.get("observing_time") else Time.now()
    )

    grid_gen = gridgen.GridGeneration(
        axes=axes,
        ctao_data_level=args_dict["ctao_data_level"],
        science_case=args_dict["science_case"],
        coordinate_system=args_dict["coordinate_system"],
        observing_location=observing_location,
        observing_time=observing_time,
    )

    grid_points = grid_gen.generate_grid()

    # Optionally convert to RA/Dec
    if args_dict["coordinate_system"] == "ra_dec":
        grid_points = grid_gen.convert_coordinates(grid_points)

    def serialize_quantity(value):
        """Serialize Quantity objects."""
        if hasattr(value, "unit"):
            return {"value": value.value, "unit": str(value.unit)}
        return value

    def clean_grid_output(grid_points, output_file=None):
        cleaned_points = []

        for point in grid_points:
            cleaned_point = {}
            for key, value in point.items():
                if isinstance(value, dict):
                    # nested dictionaries
                    cleaned_point[key] = {k: serialize_quantity(v) for k, v in value.items()}
                else:
                    cleaned_point[key] = serialize_quantity(value)

            cleaned_points.append(cleaned_point)

        output_data = json.dumps(cleaned_points, indent=4)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_data)
            _logger.info(f"Output saved to {output_file}")
        else:
            _logger.info(output_data)

    clean_grid_output(grid_points, output_file=output_filepath)


if __name__ == "__main__":
    main()
