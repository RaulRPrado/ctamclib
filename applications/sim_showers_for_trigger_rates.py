#!/usr/bin/python3

'''
    Summary
    -------
    This application simulates the cumulative PSF and compare with data (if available).

    The telescope zenith angle and the source distance can be set by command line arguments.

    The measured cumulative PSF should be provided by using the command line argument data. \
    A file name is expected, in which the file should contains 3 columns: radial distance in mm, \
    differential value of photon intensisity and its integral value.

    The MC model can be changed by providing a yaml file with the new parameter values using \
    the argument pars (see example below).

    Examples of the plots generated by this applications are shown below. On the left, \
    the cumulative PSF and on the right, the simulated PSF image.

    .. _compare_cumulative_psf_plot:
    .. image::  images/compare_cumulative_psf_North-LST-1_cumulativePSF.png
      :width: 49 %
    .. image::  images/compare_cumulative_psf_North-LST-1_image.png
      :width: 49 %

    Command line arguments
    ----------------------
    tel_name (str, required)
        Telescope name (e.g. North-LST-1, South-SST-D, ...).
    model_version (str, optional)
        Model version (default=prod4).
    src_distance (float, optional)
        Source distance in km (default=10).
    zenith (float, optional)
        Zenith angle in deg (default=20).
    data (str, optional)
        Name of the data file with the measured cumulative PSF.
    pars (str, optional)
        Yaml file with the new model parameters to replace the default ones.
    test (activation mode, optional)
        If activated, application will be faster by simulating fewer photons.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    LST-1 Prod5

    Runtime < 1 min.

    First, create an yml file named lst_pars.yml with the following content:

    .. code-block:: yaml

        mirror_reflection_random_angle: '0.0075,0.15,0.035'
        mirror_align_random_horizontal: '0.0040,28.,0.0,0.0'
        mirror_align_random_vertical: '0.0040,28.,0.0,0.0'

    And the run:

    .. code-block:: console

        python applications/compare_cumulative_psf.py --tel_name North-LST-1 \
        --model_version prod4 --pars lst_pars.yml --data PSFcurve_data_v2.txt

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
'''

import logging
import argparse

import astropy.units as u

import simtools.util.general as gen
from simtools.shower_simulator import ShowerSimulator


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Calculate and plot the PSF and eff. mirror area as a function of off-axis angle '
            'of the telescope requested.'
        )
    )
    parser.add_argument(
        '-a',
        '--array',
        help='Name of the array (e.g. 1MST, 4 LST ...)',
        type=str,
        required=True
    )
    parser.add_argument(
        '-s',
        '--site',
        help='Site name (North or South)',
        type=str,
        required=True
    )
    parser.add_argument(
        '-m',
        '--model_version',
        help='Model version (default=prod4)',
        type=str,
        default='prod4'
    )
    parser.add_argument(
        '--nruns',
        help='Number of runs (default=100)',
        type=int,
        default=100
    )
    parser.add_argument(
        '--zenith',
        help='Zenith angle in deg (default=20)',
        type=float,
        default=20
    )
    parser.add_argument(
        '--azimuth',
        help='Azimuth angle in deg (default=0)',
        type=float,
        default=0
    )
    parser.add_argument(
        '--nevents',
        help='Number of events/run (default=100)',
        type=int,
        default=100
    )
    parser.add_argument(
        '--primary',
        help='Name of the primary particle (e.g. proton, helium ...)',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output',
        help='Path of the output directory where the simulations will be saved.',
        type=str,
        default=None
    )
    parser.add_argument(
        '--test',
        help='Test option will not submit any job.',
        action='store_true'
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        dest='logLevel',
        action='store',
        default='info',
        help='Log level to print (default is INFO)'
    )

    args = parser.parse_args()
    label = 'trigger_rates'

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    showerConfigData = {
        'corsikaDataDirectory': args.output,
        'site': args.site,
        'layoutName': args.array,
        'runRange': [1, args.nruns],
        'nshow': args.nevents,
        'primary': args.primary,
        'erange': [100 * u.GeV, 300 * u.TeV],
        'eslope': -2,
        'zenith': args.zenith * u.deg,
        'azimuth': args.azimuth * u.deg,
        'viewcone': 10 * u.deg,
        'cscat': [20, 1500 * u.m, 0]
    }

    showerSimulator = ShowerSimulator(
        label=label,
        showerConfigData=showerConfigData
    )

    showerSimulator.submit(submitCommand='more ')
