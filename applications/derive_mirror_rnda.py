#!/usr/bin/python3

"""
    Summary
    -------
    This application derives the parameter mirror_reflection_random_angle \
    (mirror roughness, also called rnda here) \
    for a given set of measured photon containment diameter (e.g. 80\% containment, D80)
    of individual mirrors.  The mean value of the measured containment diameters \
    in cm is required and its sigma can be given optionally but will only be used for plotting. \

    The individual mirror focal length can be taken into account if a mirror list which contains \
    this information is used from the :ref:`Model Parameters DB` or if a new mirror list is given \
    through the argument mirror_list. Random focal lengths can be used by turning on the argument \
    use_random_focal length and a new value for it can be given through the argument random_flen.

    The algorithm works as follow: A starting value of rnda is first defined as the one taken \
    from the :ref:`Model Parameters DB` \
    (or alternatively one may want to set it using the argument rnda).\
    Secondly, ray tracing simulations are performed for single mirror configurations for each \
    mirror given in the mirror_list. The mean simulated containment diameter for all the mirrors \
    is compared with the mean measured containment diameter. A new value of rnda is then defined \
    based on the sign of the difference between measured and simulated containment diameters and a \
    new set of simulations is performed. This process repeat until the sign of the \
    difference changes, meaning that the two final values of rnda brackets the optimal. These \
    two values are used to find the optimal one by a linear \
    interpolation. Finally, simulations are performed by using the the interpolated value \
    of rnda, which is defined as the desired optimal.

    A option no_tuning can be used if one only wants to simulate one value of rnda and compare \
    the results with the measured ones.

    The results of the tuning are plotted. See examples of the containment diameter \
    D80 vs rnda plot, on the left, and the D80 distributions, on the right. \

    .. _deriva_rnda_plot:
    .. image:: images/derive_mirror_rnda_North-MST-FlashCam-D.png
      :width: 49 %
    .. image:: images/derive_mirror_rnda_North-MST-FlashCam-D_D80-distributions.png
      :width: 49 %

    This application uses the following simulation software tools:
        - sim_telarray/bin/sim_telarray
        - sim_telarray/bin/rx (optional)

    Command line arguments
    ----------------------
    telescope (str, required)
        Telescope name (e.g. North-LST-1, South-SST-D, ...)
    model_version (str, optional)
        Model version (default=prod4)
    containment_mean (float, required)
        Mean of measured containment diameter [cm]
    containment_sigma (float, optional)
        Std dev of measured containment diameter [cm]
    containment_fraction (float, required)
        Containment fractio for diameter calculation (typically 0.8)
    rnda (float, optional)
        Starting value of mirror_reflection_random_angle. If not given, the value from the \
        default model will be used.
    2f_measurement (file, optional)
        File with results from 2f measurements including mirror panel raddii and spot size \
        measurements
    d80_list (file, optional)
        File with single column list of measured D80 [cm]. It is used only for plotting the D80 \
        distributions. If given, the measured distribution will be plotted on the top of the \
        simulated one.
    mirror_list (file, optional)
        Mirror list file (in sim_telarray format) to replace the default one. It should be used \
        if measured mirror focal lengths need to be taken into account.
    use_random_flen (activation mode, optional)
        Use random focal lengths, instead of the measured ones. The argument random_flen can be \
        used to replace the default random_focal_length from the model.
    random_flen (float, optional)
        Value to replace the default random_focal_length. Only used if use_random_flen \
        is activated.
    no_tuning (activation mode, optional)
        Turn off the tuning - A single case will be simulated and plotted.
    test (activation mode, optional)
        If activated, application will be faster by simulating only few mirrors.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    MST - Prod5 (07.2020)

    Runtime about 3 min.

    .. code-block:: console

        python applications/derive_mirror_rnda.py --site North --telescope MST-FlashCam-D \
            --containment_mean 1.4 --containment_sigma 0.16 --containment_fraction 0.8 \
            --mirror_list mirror_MST_focal_lengths.dat --d80_list mirror_MST_D80.dat \
            --rnda 0.0075


    Expected output:

    .. code-block:: console

        Measured Containment Diameter (80% containment):
        Mean = 1.400 cm, StdDev = 0.160 cm

        Simulated Containment Diameter (80% containment):
        Mean = 1.401 cm, StdDev = 0.200 cm

        mirror_random_reflection_angle
        Previous value = 0.007500
        New value = 0.006378


    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
        * Fix the setStyle. For some reason, sphinx cannot built docs with it on.
"""


import logging
import matplotlib.pyplot as plt

import numpy as np
import astropy.units as u

import simtools.config as cfg
import simtools.util.general as gen
import simtools.io_handler as io
from simtools.ray_tracing import RayTracing
from simtools.model.telescope_model import TelescopeModel
import simtools.util.commandline_parser as argparser
import simtools.util.workflow_description as workflow_config
import simtools.util.model_data_writer as writer

# from simtools.visualize import setStyle

# setStyle()


def plotMeasuredDistribution(file, **kwargs):
    data = np.loadtxt(file)
    ax = plt.gca()
    ax.hist(data, **kwargs)


def parse():
    """
    Parse command line configuration

    """

    parser = argparser.CommandLineParser()
    parser.initialize_workflow_arguments()
    parser.initialize_telescope_model_arguments()
    parser.add_argument(
        "--containment_mean",
        help="Mean of measured containment diameter [cm]",
        type=float, required=True
    )
    parser.add_argument(
        "--containment_sigma",
        help="Std dev of measured containment diameter [cm]",
        type=float, required=False
    )
    parser.add_argument(
        "--containment_fraction",
        help="Containment fraction for diameter calculation (in interval 0,1)",
        type=argparser.CommandLineParser.efficiency_interval, required=False,
        default=0.8,
    )
    parser.add_argument(
        "--d80_list",
        help=(
            "File with single column list of measured D80 [cm]. If given, the measured "
            "distribution will be plotted on the top of the simulated one."
        ),
        type=str,
        required=False,
    )
    parser.add_argument(
        "--rnda",
        help="Starting value of mirror_reflection_random_angle",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--mirror_list",
        help=(
            "Mirror list file to replace the default one. It should be used if measured mirror"
            " focal lengths need to be accounted"
        ),
        type=str,
        required=False,
    )
    parser.add_argument(
        "--2f_measurement",
        help="Results from 2f measurements for each mirror panel radius and spot size",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--use_random_flen",
        help=(
            "Use random focal lengths. The argument random_flen can be used to replace the default"
            " random_focal_length parameter."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--random_flen",
        help="Value to replace the default random_focal_length.",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--no_tuning",
        help="no tuning of random_reflection_rangle - A single case will be simulated and plotted.",
        action="store_true",
    )
    parser.initialize_default_arguments()
    return parser.parse_args()


if __name__ == "__main__":

    args = parse()
    containment_fraction_percent = int(args.containment_fraction*100)
    label = "derive_mirror_rnda"

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.backends.backend_pdf').disabled = True

    workflow = workflow_config.WorkflowDescription(
        label=label,
        args=args)

    outputDir = workflow.product_data_dir

    file_writer = writer.ModelDataWriter(workflow)
    file_writer.write_metadata()
    file_writer.write_data(None)

    exit()

    tel = TelescopeModel(
        site=args.site,
        telescopeModelName=args.telescope,
        modelVersion=args.model_version,
        label=label,
    )
    if args.mirror_list is not None:
        mirrorListFile = cfg.findFile(name=args.mirror_list)
        tel.changeParameter("mirror_list", args.mirror_list)
        tel.addParameterFile("mirror_list", mirrorListFile)
    if args.random_flen is not None:
        tel.changeParameter("random_focal_length", str(args.random_flen))



    def run(rnda, plot=False):
        """Runs the simulations for one given value of rnda"""
        tel.changeParameter("mirror_reflection_random_angle", str(rnda))
        ray = RayTracing.fromKwargs(
            telescopeModel=tel,
            singleMirrorMode=True,
            mirrorNumbers=list(range(1, 10)) if args.test else "all",
            useRandomFocalLength=args.use_random_flen,
        )
        ray.simulate(test=False, force=True)  # force has to be True, always
        ray.analyze(force=True)

        # Plotting containment histograms
        if plot:
            plt.figure(figsize=(8, 6), tight_layout=True)
            ax = plt.gca()
            ax.set_xlabel(r"D$_{80}$ [cm]")

            bins = np.linspace(0.8, 3.5, 27)
            ray.plotHistogram(
                "d80_cm",
                color="r",
                linestyle="-",
                alpha=0.5,
                facecolor="r",
                edgecolor="r",
                bins=bins,
                label="simulated",
            )
            # Only plot measured D80 if the data is given
            if args.d80_list is not None:
                d80ListFile = cfg.findFile(args.d80_list)
                plotMeasuredDistribution(
                    d80ListFile,
                    color="b",
                    linestyle="-",
                    facecolor="None",
                    edgecolor="b",
                    bins=bins,
                    label="measured",
                )

            ax.legend(frameon=False)
            plotFileName = label + "_" + tel.name + "_" + "D80-distributions"
            plotFile = outputDir.joinpath(plotFileName)
            plt.savefig(str(plotFile) + ".pdf", format="pdf", bbox_inches="tight")
            plt.savefig(str(plotFile) + ".png", format="png", bbox_inches="tight")

        return (
            ray.getMean("d80_cm").to(u.cm).value,
            ray.getStdDev("d80_cm").to(u.cm).value,
        )

    # First - rnda from previous model
    if args.rnda != 0:
        rndaStart = args.rnda
    else:
        rndaStart = tel.getParameter("mirror_reflection_random_angle")["Value"]
        if isinstance(rndaStart, str):
            rndaStart = rndaStart.split()
            rndaStart = float(rndaStart[0])

    if not args.no_tuning:
        resultsRnda = list()
        resultsMean = list()
        resultsSig = list()

        def collectResults(rnda, mean, sig):
            resultsRnda.append(rnda)
            resultsMean.append(mean)
            resultsSig.append(sig)

        stop = False
        meanD80, sigD80 = run(rndaStart)
        rnda = rndaStart
        signDelta = np.sign(meanD80 - args.containment_mean)
        collectResults(rnda, meanD80, sigD80)
        while not stop:
            newRnda = rnda - (0.1 * rndaStart * signDelta)
            meanD80, sigD80 = run(newRnda)
            newSignDelta = np.sign(meanD80 - args.containment_mean)
            stop = newSignDelta != signDelta
            signDelta = newSignDelta
            rnda = newRnda
            collectResults(rnda, meanD80, sigD80)

        # Linear interpolation using two last rnda values
        resultsRnda, resultsMean, resultsSig = gen.sortArrays(
            resultsRnda, resultsMean, resultsSig
        )
        rndaOpt = np.interp(x=args.containment_mean, xp=resultsMean, fp=resultsRnda)
    else:
        rndaOpt = rndaStart

    # Running the final simulation for rndaOpt
    meanD80, sigD80 = run(rndaOpt, plot=False)

    # Printing results to stdout
    print("\nMeasured D{:}:".format(containment_fraction_percent))
    if args.containment_sigma is not None:
        print(
            "Mean = {:.3f} cm, StdDev = {:.3f} cm".format(
                args.containment_mean, args.containment_sigma)
        )
    else:
        print("Mean = {:.3f} cm".format(args.containment_mean))
    print("\nSimulated D{:}:".format(containment_fraction_percent))
    print("Mean = {:.3f} cm, StdDev = {:.3f} cm".format(meanD80, sigD80))
    print("\nmirror_random_reflection_angle")
    print("Previous value = {:.6f}".format(rndaStart))
    print("New value = {:.6f}\n".format(rndaOpt))

    # Plotting D80 vs rnda
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel(r"mirror$\_$random$\_$reflection$\_$angle")
    ax.set_ylabel(f"containment diameter $D_{{{containment_fraction_percent}}}$ [cm]")

    if not args.no_tuning:
        ax.errorbar(
            resultsRnda,
            resultsMean,
            yerr=resultsSig,
            color="k",
            marker="o",
            linestyle="none",
        )
    ax.errorbar(
        [rndaOpt],
        [meanD80],
        yerr=[sigD80],
        color="r",
        marker="o",
        linestyle="none",
        label="rnda = {:.6f} (D{:} = {:.3f} +/- {:.3f} cm)".format(
            rndaOpt, int(args.containment_fraction*100), meanD80, sigD80
        ),
    )

    xlim = ax.get_xlim()
    ax.plot(xlim, [args.containment_mean, args.containment_mean], color="k", linestyle="-")
    if args.containment_sigma is not None:
        ax.plot(
            xlim,
            [args.containment_mean + args.containment_sigma,
             args.containment_mean + args.containment_sigma],
            color="k",
            linestyle=":",
            marker=",",
        )
        ax.plot(
            xlim,
            [args.containment_mean - args.containment_sigma,
             args.containment_mean - args.containment_sigma],
            color="k",
            linestyle=":",
            marker=",",
        )

    ax.legend(frameon=False, loc="upper left")

    plotFileName = label + "_" + tel.name
    plotFile = outputDir.joinpath(plotFileName)
    plt.savefig(str(plotFile) + ".pdf", format="pdf", bbox_inches="tight")
    plt.savefig(str(plotFile) + ".png", format="png", bbox_inches="tight")
