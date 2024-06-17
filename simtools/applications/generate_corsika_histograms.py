#!/usr/bin/python3

"""
    Summary
    -------
    This application generates a set of histograms of the distribution of Cherenkov photons on the
    ground (at observation level) read from the CORSIKA IACT output file provided as input.

    The histograms can be saved both into pdfs and in a hdf5 file.

    The following 2D histograms are generated:
        - Number of Cherenkov photons on the ground;
        - Density of Cherenkov photons on the ground;
        - Incoming direction (directive cosines) of the Cherenkov photons;
        - Time of arrival (ns) vs altitude of production (km);
        - Number of Cherenkov photons per event per telescope.

    The following 1D histograms are generated:
        - Wavelength;
        - Counts;
        - Density;
        - Time of arrival;
        - Altitude of production;
        - Number of photons per telescope;
        - Number of photons per event.

    Histograms for the distribution of CORSIKA event header elements can also be generated by using
    the --event_1d_histograms and --event_2d_histograms arguments. The accepted arguments (keys)
    are to be found in the CORSIKA manual, e.g., "total_energy", "zenith", "azimuth".

    Command line arguments
    ----------------------
    iact_file (str, required)
        The name of the CORSIKA IACT file resulted from the CORSIKA simulation.

    telescope_indices (list, optional)
        The list with the telescope indices to be considered in the generation of the histograms.
        Telescopes that are not in the list will not contribute with photons to the histograms.
        If the argument is not given, all telescopes are considered.

    individual_telescopes (bool, optional)
        Indicates whether single histograms are generated for the individual telescopes, or if
        a master histogram is generated for all the telescopes together.
        If the argument is not given, the Cherenkov photons from the given telescopes are considered
        together in the same histograms.

    hist_config (hdf5 or dict, optional)
        The configuration used for generating the histograms.
        It includes information about the bin sizes, the ranges, scale of the plot and units.
        By construction, three major histograms are created to start with:
        - hist_direction (2D): Directive cosines (x and y) for the incoming photons;
        - hist_position (3D): position x, position y, and wavelength;
        - hist_time_altitude (2D): time of arrival and altitude of emission;

    If the argument is not given, the default configuration is generated:

    .. code-block:: console

        hist_direction:
            x axis: {bins: 100, scale: linear, start: -1, stop: 1}
            y axis: {bins: 100, scale: linear, start: -1, stop: 1}

        hist_position:
            x axis:
                bins: 100
                scale: linear
                start: !astropy.units.Quantity
                    unit: &id001 m
                    value: -1000.0
                stop: &id002 !astropy.units.Quantity
                    unit: *id001
                    value: 1000.0
            y axis:
                bins: 100
                scale: linear
                start: !astropy.units.Quantity
                    unit: *id001
                    value: -1000.0
                stop: *id002
            z axis:
                bins: 80
                scale: linear
                start: !astropy.units.Quantity
                    unit: nm
                    value: 200.0
                stop: !astropy.units.Quantity
                    unit: *id003
                    value: 1000.0
        hist_time_altitude:
            x axis:
                bins: 100
                scale: linear
                start: !astropy.units.Quantity
                    unit: ns
                    value: -2000.0
                stop: !astropy.units.Quantity
                    unit: *id004
                    value: 2000.0
            y axis:
                bins: 100
                scale: linear
                start: !astropy.units.Quantity
                    unit: km
                    value: 120.0
                stop: !astropy.units.Quantity
                    unit: *id005
                    value: 0.0


    pdf (bool, optional)
        If set, histograms are saved into pdf files.
        One pdf file contains all the histograms for the Cherenkov photons.
        The name of the file is controlled via hdf5_file_name.
        If event_1d_histograms and event_2d_histograms are used, two separate pdf files might be
        created to accommodate the histograms for the CORSIKA event header elements. The core names
        of these output pdf files are also given by hdf5_file_name argument with the addition of
        'event_1d_histograms' and 'event_2d_histograms'.


    hdf5 (bool, optional)
        If set, histograms are saved into hdf5 files.

    hdf5_file_name (str, optional)
        The name of the output hdf5 data (without the path).
        It requires the --hdf5 flag.
        If not given, hdf5_file_name takes the name from the input IACT file (input_file).
        If the output hdf5_file_name file already exists, the tables associated to the chosen
        flags (e.g. hdf5, event_1d_histograms, event_2d_histograms) will be overwritten. The
        remaining tables, if any, will stay untouched.


    event_1d_histograms (str, optional)
        Generate 1D histograms for elements given in --event_1d_histograms from the CORSIKA event
        header and save into hdf5/pdf files.
        It allows more than one argument, separated by simple spaces.
        Usage: --event_1d_histograms first_interaction_height total_energy.

    event_2d_histograms (str, optional)
        Generate 2D histograms for elements given in --event_2d_histograms from the CORSIKA event
        header and save into hdf5/pdf files.
        It allows more than one argument, separated by simple spaces.
        The elements are grouped into pairs and the 2D histograms are generated always for two
        subsequent elements.
        For example, --event_2d_histograms first_interaction_height total_energy zenith azimuth
        will generate one 2D histogram for first_interaction_height total_energy and another 2D
        histogram for zenith and azimuth.

    Example
    -------
    Generate the histograms for a test IACT file:

     .. code-block:: console

        simtools-generate-corsika-histograms --iact_file /workdir/external/simtools/tests/\
            resources/tel_output_10GeV-2-gamma-20deg-CTAO-South.corsikaio --pdf --hdf5
            --event_2d_histograms zenith azimuth --event_1d_histograms total_energy


    Expected final print-out message:

    .. code-block:: console

        INFO::generate_corsika_histograms(l358)::main::Finalizing the application.
        Total time needed: 8s.
"""

import logging
import re
import time
from pathlib import Path

import numpy as np

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.corsika import corsika_histograms_visualize
from simtools.corsika.corsika_histograms import CorsikaHistograms
from simtools.io_operations import io_handler

logger = logging.getLogger()


def _parse(label, description):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing the application.
    description: str
        Description of the application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--iact_file",
        help="Name of the CORSIKA IACT file from which to generate the histograms.",
        type=str,
        required=True,
    )

    config.parser.add_argument(
        "--telescope_indices",
        help="Name of the CORSIKA IACT file from which to generate the histograms.",
        type=str,
        required=False,
        nargs="+",
        default=None,
    )

    config.parser.add_argument(
        "--individual_telescopes",
        help="if False, the histograms are filled for all given telescopes together, otherwise"
        "one histogram is set for each telescope separately.",
        action="store_true",
        required=False,
        default=False,
    )

    config.parser.add_argument(
        "--hist_config",
        help="hdf5 file with the configuration parameters to create the histograms.",
        type=str,
        required=False,
        default=None,
    )

    config.parser.add_argument(
        "--pdf", help="Save histograms into a pdf file.", action="store_true", required=False
    )

    config.parser.add_argument(
        "--hdf5", help="Save histograms into hdf5 files.", action="store_true", required=False
    )

    config.parser.add_argument(
        "--hdf5_file_name",
        help="Name of the hdf5 file where to save the histograms.",
        type=str,
        required=False,
        default=None,
    )

    config.parser.add_argument(
        "--event_1d_histograms",
        help="The keys from the CORSIKA event header to be used for the generation of 1D "
        "histograms. The available choices can been found in the all_event_keys attribute of"
        "the CorsikaHistograms.",
        required=False,
        default=None,
        nargs="*",
    )

    config.parser.add_argument(
        "--event_2d_histograms",
        help="The keys from the CORSIKA event header to be used for the generation of 2D "
        "histograms. The available choices can been found in the all_event_keys attribute of"
        "the CorsikaHistograms.",
        required=False,
        default=None,
        nargs="*",
    )

    config_parser, _ = config.initialize(db_config=False, paths=True)

    if (
        not config_parser["pdf"]
        and not config_parser["hdf5"]
        and not config_parser["event_1d_histograms"]
        and not config_parser["event_2d_histograms"]
    ):
        config.parser.error(
            "At least one argument is required: --pdf, --hdf5, --event_1d_histograms, or "
            "--event_2d_histograms."
        )

    return config_parser, _


def _plot_figures(corsika_histograms_instance, test=False):
    """
    Auxiliary function to centralize the plotting functions.

    Parameters
    ----------
    corsika_histograms_instance: CorsikaHistograms instance.
        The CorsikaHistograms instance created in main.
    test: bool
        If true plots the figures for the first two functions only.
    """

    plot_function_names = [
        plotting_method
        for plotting_method in dir(corsika_histograms_visualize)
        if plotting_method.startswith("plot_")
        and "event_header_distribution" not in plotting_method
    ]
    if test:
        plot_function_names = plot_function_names[:2]

    figure_list = []
    for function_name in plot_function_names:
        plot_function = getattr(corsika_histograms_visualize, function_name)
        figures = plot_function(corsika_histograms_instance)
        for fig in figures:
            figure_list.append(fig)

    figure_list = np.array(figure_list).flatten()
    core_name = re.sub(r"\.hdf5$", "", corsika_histograms_instance.hdf5_file_name)
    output_file_name = Path(corsika_histograms_instance.output_path).joinpath(f"{core_name}.pdf")
    corsika_histograms_visualize.save_figs_to_pdf(figure_list, output_file_name)


def _derive_event_1d_histograms(
    corsika_histograms_instance, event_1d_header_keys, pdf, hdf5, overwrite=False
):
    """
    Auxiliary function to derive the histograms for the arguments given by event_1d_histograms.

    Parameters
    ----------
    corsika_histograms_instance: CorsikaHistograms instance.
        The CorsikaHistograms instance created in main.
    event_1d_header_keys: str
        Generate 1D histograms for elements given in event_1d_header_keys from the CORSIKA event
        header and save into hdf5/pdf files.
    pdf: bool
        If true, histograms are saved into a pdf file.
    hdf5: bool
        If true, histograms are saved into hdf5 files.
    overwrite: bool
        If true, overwrites the current output hdf5 file.
    """
    figure_list = []
    for event_header_element in event_1d_header_keys:
        if pdf:
            figure = corsika_histograms_visualize.plot_1d_event_header_distribution(
                corsika_histograms_instance, event_header_element
            )
            figure_list.append(figure)
        if hdf5:
            corsika_histograms_instance.export_event_header_1d_histogram(
                event_header_element, bins=50, hist_range=None, overwrite=overwrite
            )
    if pdf:
        figures_list = np.array(figure_list).flatten()
        output_file_name = Path(corsika_histograms_instance.output_path).joinpath(
            f"{corsika_histograms_instance.hdf5_file_name}_event_1d_histograms.pdf"
        )
        corsika_histograms_visualize.save_figs_to_pdf(figures_list, output_file_name)


def _derive_event_2d_histograms(
    corsika_histograms_instance, event_2d_header_keys, pdf, hdf5, overwrite=False
):
    """
    Auxiliary function to derive the histograms for the arguments given by event_1d_histograms.
    If an odd number of event header keys are given, the last one is discarded.

    Parameters
    ----------
    corsika_histograms_instance: CorsikaHistograms instance.
        The CorsikaHistograms instance created in main.
    event_2d_header_keys: str
        Generate 1D histograms for elements given in event_1d_header_keys from the CORSIKA event
        header and save into hdf5/pdf files.
    pdf: bool
        If true, histograms are saved into a pdf file.
    hdf5: bool
        If true, histograms are saved into hdf5 files.
    overwrite: bool
        If true, overwrites the current output hdf5 file.
    """
    figure_list = []
    for i_event_header_element, _ in enumerate(event_2d_header_keys[::2]):
        # [::2] to discard the last one in case an odd number of keys are passed

        if len(event_2d_header_keys) % 2 == 1:  # if odd number of keys
            msg = (
                "An odd number of keys was passed to generate 2D histograms."
                "The last key is being ignored."
            )
            logger.warning(msg)

        if pdf:
            figure = corsika_histograms_visualize.plot_2d_event_header_distribution(
                corsika_histograms_instance,
                event_2d_header_keys[i_event_header_element],
                event_2d_header_keys[i_event_header_element + 1],
            )
            figure_list.append(figure)
        if hdf5:
            corsika_histograms_instance.export_event_header_2d_histogram(
                event_2d_header_keys[i_event_header_element],
                event_2d_header_keys[i_event_header_element + 1],
                bins=50,
                hist_range=None,
                overwrite=overwrite,
            )
    if pdf:
        figures_list = np.array(figure_list).flatten()
        output_file_name = Path(corsika_histograms_instance.output_path).joinpath(
            f"{corsika_histograms_instance.hdf5_file_name}_event_2d_histograms.pdf"
        )
        corsika_histograms_visualize.save_figs_to_pdf(figures_list, output_file_name)


def main():
    label = Path(__file__).stem
    description = "Generate histograms for the Cherenkov photons saved in the CORSIKA IACT file."
    io_handler_instance = io_handler.IOHandler()
    args_dict, _ = _parse(label, description)

    output_path = io_handler_instance.get_output_directory(label, sub_dir="application-plots")

    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    initial_time = time.time()
    logger.info("Starting the application.")

    corsika_histograms_instance = CorsikaHistograms(
        args_dict["iact_file"], output_path=output_path, hdf5_file_name=args_dict["hdf5_file_name"]
    )
    if args_dict["telescope_indices"] is not None:
        try:
            indices = np.array(args_dict["telescope_indices"]).astype(int)
        except ValueError:
            msg = (
                f"{args_dict['telescope_indices']} not a valid input. "
                f"Please use integer numbers for telescope_indices"
            )
            logger.error(msg)
            raise
    else:
        indices = None
    # If the hdf5 output file already exists, the results are appended to it.
    if (Path(corsika_histograms_instance.hdf5_file_name).exists()) and (
        args_dict["hdf5"] or args_dict["event_1d_histograms"] or args_dict["event_2d_histograms"]
    ):
        msg = (
            f"Output hdf5 file {corsika_histograms_instance.hdf5_file_name} already exists. "
            f"Overwriting it."
        )
        logger.warning(msg)
        overwrite = True
    else:
        overwrite = False
    corsika_histograms_instance.set_histograms(
        telescope_indices=indices,
        individual_telescopes=args_dict["individual_telescopes"],
        hist_config=args_dict["hist_config"],
    )

    # Cherenkov photons
    if args_dict["pdf"]:
        _plot_figures(
            corsika_histograms_instance=corsika_histograms_instance, test=args_dict["test"]
        )
    if args_dict["hdf5"]:
        corsika_histograms_instance.export_histograms(overwrite=overwrite)

    # Event information
    if args_dict["event_1d_histograms"] is not None:
        _derive_event_1d_histograms(
            corsika_histograms_instance,
            args_dict["event_1d_histograms"],
            args_dict["pdf"],
            args_dict["hdf5"],
            overwrite=not args_dict["hdf5"],
        )
    if args_dict["event_2d_histograms"] is not None:
        _derive_event_2d_histograms(
            corsika_histograms_instance,
            args_dict["event_2d_histograms"],
            args_dict["pdf"],
            args_dict["hdf5"],
            overwrite=not (args_dict["hdf5"] or args_dict["event_1d_histograms"]),
        )

    final_time = time.time()
    logger.info(
        f"Finalizing the application. Total time needed: {round(final_time - initial_time)}s."
    )


if __name__ == "__main__":
    main()
