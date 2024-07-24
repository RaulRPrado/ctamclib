#!/usr/bin/python3

r"""
    Validate the camera efficiency by simulating it using the sim_telarray testeff program.

    The results of camera efficiency for Cherenkov (left) and NSB light (right) as a function\
    of wavelength are plotted. See examples below.

    .. _validate_camera_eff_plot:
    .. image:: images/validate_camera_efficiency_North-MST-NectarCam-D_cherenkov.png
      :width: 49 %
    .. image:: images/validate_camera_efficiency_North-MST-NectarCam-D_nsb.png
      :width: 49 %

    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LSTN-01, SSTS-15)
    model_version (str, optional)
        Model version
    zenith_angle (float, optional)
        Zenith angle in degrees (between 0 and 180).
    azimuth_angle (float, optional)
        Telescope pointing direction in azimuth.
    nsb_spectrum (str, optional)
        File with NSB spectrum to use for the efficiency simulation.

    Example
    -------
    MSTN-01 Prod5

    Runtime < 1 min.

    .. code-block:: console

        simtools-validate-camera-efficiency --site North \\
            --azimuth_angle 0 --zenith_angle 20 \\
            --nsb_spectrum average_nsb_spectrum_CTAO-N_ze20_az0.txt \\
            --telescope MSTN-01 --model_version prod5

    The output is saved in simtools-output/validate_camera_efficiency.
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.camera_efficiency import CameraEfficiency
from simtools.configuration import configurator


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=(
            "Calculate the camera efficiency of the telescope requested. "
            "Plot the camera efficiency vs wavelength for cherenkov and NSB light."
        ),
    )
    config.parser.add_argument(
        "--nsb_spectrum",
        help=(
            "File with NSB spectrum to use for the efficiency simulation."
            "The expected format is two columns with wavelength in nm and "
            "NSB flux with the units: [1e9 * ph/m2/s/sr/nm]."
            "If the file has more than two columns, the first and third are used,"
            "and the second is ignored (native sim_telarray behavior)."
        ),
        type=str,
        default=None,
        required=False,
    )
    _args_dict, _db_config = config.initialize(
        db_config=True,
        simulation_model="telescope",
        simulation_configuration={"corsika_configuration": ["zenith_angle", "azimuth_angle"]},
    )
    if _args_dict["site"] is None or _args_dict["telescope"] is None:
        config.parser.print_help()
        print("\n\nSite and telescope must be provided\n\n")
        raise RuntimeError("Site and telescope must be provided")
    return _args_dict, _db_config


def main():  # noqa: D103
    label = Path(__file__).stem
    args_dict, _db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    ce = CameraEfficiency(
        db_config=_db_config,
        simtel_path=args_dict["simtel_path"],
        label=label,
        config_data=args_dict,
    )
    ce.simulate()
    ce.analyze(force=True)
    ce.plot_efficiency(efficiency_type="Cherenkov", save_fig=True)
    ce.plot_efficiency(efficiency_type="NSB", save_fig=True)


if __name__ == "__main__":
    main()
