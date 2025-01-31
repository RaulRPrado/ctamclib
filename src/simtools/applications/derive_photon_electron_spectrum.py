#!/usr/bin/python3

r"""
    Derive single photon electron spectrum from a given amplitude spectrum.

    Normalizes singe-p.e. amplitude distribution to mean amplitude of 1.0,
    as required by sim_telarray. Allows to fold in afterpulse distribution
    to prompt spectrum.

    Input files can be in ecsv format (preferred) or in sim_telarray legacy format.

    Three output files are written to the output directory:

    - 'output_file'.ecsv: Single photon electron spectrum in ecsv format (data and metadata).
    - 'output_file'.dat: Single photon electron spectrum in sim_telarray format.

    Example
    -------

    .. code-block:: console

        simtools-derive-photon-electron-spectrum \\
            --input_spectrum spectrum_photon_electron.ecsv \\
            --afterpulse_spectrum spectrum_afterpulse.ecsv \\
            --step_size 0.02 \\
            --max_amplitude 42.0 \\
            --use_norm_spe \\
            --output_path ./tests/output \\
            --output_file spectrum_photon_electron_afterpulse.ecsv

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.camera.single_photon_electron_spectrum import SinglePhotonElectronSpectrum
from simtools.configuration import configurator


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description="Derive single photon electron spectrum from a given amplitude spectrum.",
    )
    config.parser.add_argument(
        "--input_spectrum",
        help="File with amplitude spectrum.",
        type=Path,
        required=True,
    )
    config.parser.add_argument(
        "--afterpulse_spectrum",
        help="File with afterpulse spectrum.",
        type=Path,
        required=False,
    )
    config.parser.add_argument(
        "--step_size",
        help="Step size in amplitude spectrum",
        type=float,
        default=0.02,
        required=False,
    )
    config.parser.add_argument(
        "--max_amplitude",
        help="Maximum amplitude in amplitude spectrum",
        type=float,
        default=42.0,
        required=False,
    )
    config.parser.add_argument(
        "--use_norm_spe",
        help="Use sim_telarray tool 'norm_spe' to normalize the spectrum.",
        action="store_true",
        required=False,
    )

    return config.initialize(db_config=False, output=True)


def main():  # noqa: D103
    args_dict, _ = _parse(Path(__file__).stem)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    single_pe = SinglePhotonElectronSpectrum(args_dict)
    single_pe.derive_single_pe_spectrum()
    single_pe.write_single_pe_spectrum()


if __name__ == "__main__":
    main()
