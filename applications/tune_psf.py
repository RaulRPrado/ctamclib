#!/usr/bin/python3

"""
    Summary
    -------
    This applications tunes the parameters mirror_reflection_random_angle, \
    mirror_align_random_horizontal and mirror_align_random_vertical using \
    cumulative PSF measurement.

    The telescope zenith angle and the source distance can be set by command line arguments.

    The measured cumulative PSF should be provided by using the command line argument data. \
    A file name is expected, in which the file should contains 3 columns: radial distance in mm, \
    differential value of photon intensity and its integral value.

    The tuning is performed through a random search. A number of random combination of the \
    parameters are tested and the best ones are selected based on the minimum value of \
    the Root Mean Squared Deviation between data and simulations. The range in which the \
    parameter are drawn uniformly are defined based on the previous value on the telescope model.

    The assumption are:

    a) mirror_align_random_horizontal and mirror_align_random_vertical are the same.

    b) mirror_align_random_horizontal/vertical have no dependence on the zneith angle.

    One example of the plot generated by this applications are shown below.

    .. _tune_psf_plot:
    .. image::  images/tune_psf.png
      :width: 49 %

    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LST-1, SST-D, ...).
    model_version (str, optional)
        Model version (default=prod4).
    src_distance (float, optional)
        Source distance in km (default=10).
    zenith (float, optional)
        Zenith angle in deg (default=20).
    data (str, optional)
        Name of the data file with the measured cumulative PSF.
    plot_all (activation mode, optional)
        If activated, plots will be generated for all values tested during tuning.
    fixed (activation mode, optional)
        Keep the first entry of mirror_reflection_random_angle fixed.
    test (activation mode, optional)
        If activated, application will be faster by simulating fewer photons.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    LST-1 Prod5

    Runtime around 5 min.

    .. code-block:: console

        python applications/tune_psf.py --site North --telescope LST-1 \
            --model_version prod5 --data PSFcurve_data_v2.txt --plot_all
"""

import logging
from collections import OrderedDict

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools import io_handler, visualize
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing
from simtools.util.model import split_simtel_parameter


def load_data(datafile):
    d_type = {"names": ("Radius [cm]", "Cumulative PSF"), "formats": ("f8", "f8")}
    data = np.loadtxt(datafile, dtype=d_type, usecols=(0, 2))
    data["Radius [cm]"] *= 0.1
    data["Cumulative PSF"] /= np.max(np.abs(data["Cumulative PSF"]))
    return data


def main():

    config = configurator.Configurator(
        description=(
            "Tune mirror_reflection_random_angle, mirror_align_random_horizontal "
            "and mirror_align_random_vertical using cumulative PSF measurement."
        )
    )
    config.parser.add_argument(
        "--src_distance",
        help="Source distance in km (default=10)",
        type=float,
        default=10,
    )
    config.parser.add_argument(
        "--zenith", help="Zenith angle in deg (default=20)", type=float, default=20
    )
    config.parser.add_argument(
        "--data", help="Data file name with the measured PSF vs radius [cm]", type=str
    )
    config.parser.add_argument(
        "--plot_all",
        help=(
            "On: plot cumulative PSF for all tested combinations, "
            "Off: plot it only for the best set of values"
        ),
        action="store_true",
    )
    config.parser.add_argument(
        "--fixed",
        help=("Keep the first entry of mirror_reflection_random_angle fixed."),
        action="store_true",
    )

    args_dict, db_config = config.initialize(db_config=True, telescope_model=True)
    label = "tune_psf"

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label, dir_type="application-plots")

    tel_model = TelescopeModel(
        site=args_dict["site"],
        telescope_model_name=args_dict["telescope"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )
    # If we want to start from values different than the ones currently in the model:
    # align = 0.0046
    # pars_to_change = {
    #     'mirror_reflection_random_angle': '0.0075 0.125 0.0037',
    #     'mirror_align_random_horizontal': f'{align} 28 0 0',
    #     'mirror_align_random_vertical': f'{align} 28 0 0',
    # }
    # tel_model.change_multiple_parameters(**pars_to_change)

    all_parameters = list()

    def add_parameters(
        mirror_reflection, mirror_align, mirror_reflection_fraction=0.15, mirror_reflection_2=0.035
    ):
        """
        Transform the parameters to the proper format and add a new set of
        parameters to the all_parameters list.
        """
        pars = dict()
        mrra = "{:.4f},{:.2f},{:.4f}".format(
            mirror_reflection, mirror_reflection_fraction, mirror_reflection_2
        )
        pars["mirror_reflection_random_angle"] = mrra
        mar = "{:.4f},28.,0.,0.".format(mirror_align)
        pars["mirror_align_random_horizontal"] = mar
        pars["mirror_align_random_vertical"] = mar
        all_parameters.append(pars)

    # Grabbing the previous values of the parameters from the tel model.
    #
    # mrra -> mirror reflection random angle (first entry of mirror_reflection_random_angle)
    # mfr -> mirror fraction random (second entry of mirror_reflection_random_angle)
    # mrra2 -> mirror reflection random angle 2 (third entry of mirror_reflection_random_angle)
    # mar -> mirror align random (first entry of mirror_align_random_horizontal/vertical)

    raw_par = tel_model.get_parameter("mirror_reflection_random_angle")["Value"]
    split_par = split_simtel_parameter(raw_par)
    mrra_0 = split_par[0]
    mfr_0 = split_par[1]
    mrra2_0 = split_par[2]

    raw_par = tel_model.get_parameter("mirror_align_random_horizontal")["Value"]
    mar_0 = split_simtel_parameter(raw_par)[0]

    logger.debug(
        "Previous parameter values: \n"
        "MRRA = " + str(mrra_0) + "\n"
        "MRF = " + str(mfr_0) + "\n"
        "MRRA2 = " + str(mrra2_0) + "\n"
        "MAR = " + str(mar_0) + "\n"
    )

    if args_dict["fixed"]:
        logger.debug("fixed=True - First entry of mirror_reflection_random_angle is kept fixed.")

    # Drawing parameters randonly
    # Range around the previous values are hardcoded
    # Number of runs is hardcoded
    N_RUNS = 50
    for _ in range(N_RUNS):
        mrra_range = 0.004 if not args_dict["fixed"] else 0
        mrf_range = 0.1
        mrra2_range = 0.03
        mar_range = 0.005
        mrra = np.random.uniform(max(mrra_0 - mrra_range, 0), mrra_0 + mrra_range)
        mrf = np.random.uniform(max(mfr_0 - mrf_range, 0), mfr_0 + mrf_range)
        mrra2 = np.random.uniform(max(mrra2_0 - mrra2_range, 0), mrra2_0 + mrra2_range)
        mar = np.random.uniform(max(mar_0 - mar_range, 0), mar_0 + mar_range)
        add_parameters(mrra, mar, mrf, mrra2)

    # Loading measured cumulative PSF
    data_to_plot = OrderedDict()
    if args_dict["data"] is not None:
        data_file = gen.find_file(args_dict["data"], args_dict["model_path"])
        data_to_plot["measured"] = load_data(data_file)
        radius = data_to_plot["measured"]["Radius [cm]"]

    # Preparing figure name
    plot_file_name = "_".join((label, tel_model.name + ".pdf"))
    plot_file = output_dir.joinpath(plot_file_name)
    pdf_pages = PdfPages(plot_file)

    def calculate_RMSD(data, sim):
        """
        Calculates the Root Mean Squared Deviation to be used
        as metric to find the best parameters.
        """
        return np.sqrt(np.mean((data - sim) ** 2))

    def run_pars(pars, plot=True):
        """
        Runs the tuning for one set of parameters, add a plot to the pdfPages
        (if plot=True) and returns the RMSD and the D80.
        """
        tel_model.change_multiple_parameters(**pars)

        ray = RayTracing.from_kwargs(
            telescope_model=tel_model,
            simtel_source_path=args_dict["simtelpath"],
            source_distance=args_dict["src_distance"] * u.km,
            zenith_angle=args_dict["zenith"] * u.deg,
            off_axis_angle=[0.0 * u.deg],
        )

        ray.simulate(test=args_dict["test"], force=True)
        ray.analyze(force=True, use_rx=False)

        # Plotting cumulative PSF
        im = ray.images()[0]
        d80 = im.get_psf()

        # Simulated cumulative PSF
        data_to_plot["simulated"] = im.get_cumulative_data(radius * u.cm)

        rmsd = calculate_RMSD(
            data_to_plot["measured"]["Cumulative PSF"], data_to_plot["simulated"]["Cumulative PSF"]
        )

        if plot:
            fig = visualize.plot_1D(
                data_to_plot,
                plot_difference=True,
                no_markers=True,
            )
            ax = fig.get_axes()[0]
            ax.set_ylim(0, 1.05)
            ax.set_title(
                "refl_rnd={}, align_rnd={}".format(
                    pars["mirror_reflection_random_angle"], pars["mirror_align_random_vertical"]
                )
            )

            ax.text(
                0.8,
                0.3,
                "D80 = {:.3f} cm\nRMSD = {:.4f}".format(d80, rmsd),
                verticalalignment="center",
                horizontalalignment="center",
                transform=ax.transAxes,
            )
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.clf()

        return d80, rmsd

    # Running the tuning for all parameters in all_parameters
    # and storing the best parameters in best_pars
    min_rmsd = 100
    for pars in all_parameters:
        _, rmsd = run_pars(pars, plot=args_dict["plot_all"])
        if rmsd < min_rmsd:
            min_rmsd = rmsd
            best_pars = pars

    # Rerunnig and plotting the best pars
    run_pars(best_pars, plot=True)

    plt.close()
    pdf_pages.close()

    # Printing the results
    print("Best parameters:")
    for par, value in best_pars.items():
        print("{} = {}".format(par, value))


if __name__ == "__main__":
    main()
