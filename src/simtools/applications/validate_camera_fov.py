#!/usr/bin/python3

r"""
    Calculate the camera FoV of the telescope requested and plot the camera.

    The orientation for the plotting is "as seen for an observer facing the camera".

    An example of the camera plot can be found below.

    .. _camera_fov_plot:
    .. image:: images/validate_camera_fov_North-LST-1_pixelLayout.png
      :width: 50 %


    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LST-1, SST-D, ...)
    model_version (str, optional)
        Model version
    camera_in_sky_coor (bool, optional)
        Plot the camera layout in sky coordinates akin to looking at it from behind for single \
         mirror telescopes
    print_pixels_id (bool, optional)
        Up to which pixel ID to print. To suppress printing of pixel IDs, set to zero\
         (--print_pixels_id 0). To print all pixels, set to 'All'."

    Example
    -------
    LST - 5.0.0

    .. code-block:: console

        simtools-validate-camera-fov --site North \\
            --telescope LSTN-01 --model_version 5.0.0

    The output is saved in simtools-output/validate_camera_fov.

    Expected final print-out message:

    .. code-block:: console

        Saved camera plot in /workdir/external/simtools/simtools-output/validate_camera_fov\\
        /application-plots/validate_camera_fov_LST-1_pixel_layout.png

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.visualization import plot_camera, visualize


def _parse():
    config = configurator.Configurator(
        label=Path(__file__).stem,
        description=(
            "Calculate the camera FoV of the telescope requested. "
            "Plot the camera, as seen for an observer facing the camera."
        ),
    )
    config.parser.add_argument(
        "--camera_in_sky_coor",
        help=(
            "Plot the camera layout in sky coordinates "
            "(akin to looking at it from behind for single mirror telescopes)"
        ),
        action="store_true",
        default=False,
    )
    config.parser.add_argument(
        "--print_pixels_id",
        help=(
            "Up to which pixel ID to print. "
            "To suppress printing of pixel IDs, set to zero (--print_pixels_id 0). "
            "To print all pixels, set to 'All'."
        ),
        default=50,
    )
    return config.initialize(db_config=True, simulation_model="telescope")


def main():  # noqa: D103
    args_dict, db_config = _parse()

    label = "validate_camera_fov"

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label, sub_dir="application-plots")

    tel_model = TelescopeModel(
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )
    tel_model.export_model_files()

    print(f"\nValidating the camera FoV of {tel_model.name}\n")

    focal_length = tel_model.get_telescope_effective_focal_length("cm")
    camera = tel_model.camera

    fov, r_edge_avg = camera.calc_fov()

    print("\nEffective focal length = " + f"{focal_length:.3f} cm")
    print(f"{tel_model.name} FoV = {fov:.3f} deg")
    print(f"Avg. edge radius = {r_edge_avg:.3f} cm\n")

    # Now plot the camera as well
    try:
        pixel_ids_to_print = int(args_dict["print_pixels_id"])
        if pixel_ids_to_print == 0:
            pixel_ids_to_print = -1  # so not print the zero pixel
    except ValueError as exc:
        if args_dict["print_pixels_id"].lower() == "all":
            pixel_ids_to_print = camera.get_number_of_pixels()
        else:
            raise ValueError(
                f"The value provided to --print_pixels_id ({args_dict['print_pixels_id']}) "
                "should be an integer or All"
            ) from exc
    fig = plot_camera.plot_pixel_layout(camera, args_dict["camera_in_sky_coor"], pixel_ids_to_print)
    plot_file_prefix = output_dir.joinpath(f"{label}_{tel_model.name}_pixel_layout")
    visualize.save_figure(fig, f"{plot_file_prefix!s}", log_title="camera")


if __name__ == "__main__":
    main()
