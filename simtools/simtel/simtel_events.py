import math
import logging
import numpy as np
from copy import copy

import astropy.units as u

from eventio.simtel import SimTelFile


__all__ = ['SimtelEvents']


class InconsistentInputFile(Exception):
    pass


class SimtelEvents:
    '''
    This class handle sim_telarray events.
    sim_telarray files are read with eventio package,

    Methods
    -------
    plotAndSaveFigures(figName)
        Plot all histograms and save a single pdf file.


    Attributes
    ----------
    inputFiles: list
        List of sim_telarray files.
    summaryEvents: dict
        Arrays of energy and core radius of events.
    '''

    def __init__(self, inputFiles=None):
        '''
        SimtelEvents

        Parameters
        ----------
        inputFiles: list
            List of sim_telarray output files (str of Path).
        '''
        self._logger = logging.getLogger(__name__)
        self.loadInputFiles(inputFiles)
        if self.numberOfFiles > 0:
            self.loadHeaderAndSummary()

    def loadInputFiles(self, files=None):
        '''
        Store list of input files into inputFiles attribute.

        Parameters
        ----------
        files: list
            List of sim_telarray files (str or Path).
        '''
        if not hasattr(self, 'inputFiles'):
            self.inputFiles = list()

        if files is None:
            msg = 'No input file was given'
            self._logger.debug(msg)
            return

        if not isinstance(files, list):
            files = [files]

        for file in files:
            self.inputFiles.append(file)
        return

    @property
    def numberOfFiles(self):
        ''' Number of files loaded. '''
        return len(self.inputFiles) if hasattr(self, 'inputFiles') else 0

    def loadHeaderAndSummary(self):
        '''
        Read MC header from sim_telarray files and store it into _mcHeader.
        Also fills summaryEvents with energy and core radius of triggered events.
        '''

        self._numberOfFiles = len(self.inputFiles)
        keysToGrab = ['obsheight', 'n_showers', 'n_use', 'core_range', 'diffuse', 'viewcone',
                      'E_range', 'spectral_index', 'B_total']
        self._mcHeader = dict()

        def _areHeadersConsistent(header0, header1):
            comparison = dict()
            for k in keysToGrab:
                value = (header0[k] == header1[k])
                comparison[k] = value if isinstance(value, bool) else all(value)

            return all(comparison)

        isFirstFile = True
        numberOfTriggeredEvents = 0
        summaryEnergy, summaryRcore = list(), list()
        for file in self.inputFiles:
            with SimTelFile(file) as f:

                for event in f:
                    en = event['mc_shower']['energy']
                    rc = math.sqrt(
                        math.pow(event['mc_event']['xcore'], 2)
                        + math.pow(event['mc_event']['ycore'], 2)
                    )

                    summaryEnergy.append(en)
                    summaryRcore.append(rc)
                    numberOfTriggeredEvents += 1

                if isFirstFile:
                    # First file - grabbing parameters
                    self._mcHeader.update({k: copy(f.mc_run_headers[0][k]) for k in keysToGrab})
                else:
                    # Remaining files - Checking whether the parameters are consistent
                    if not _areHeadersConsistent(self._mcHeader, f.mc_run_headers[0]):
                        msg = 'MC header pamameters from different files are inconsistent'
                        self._logger.error(msg)
                        raise InconsistentInputFile(msg)

                isFirstFile = False

        self.summaryEvents = {
            'energy': np.array(summaryEnergy),
            'r_core': np.array(summaryRcore)
        }

        # Calculating number of events
        self._mcHeader['n_events'] = (
            self._mcHeader['n_use'] * self._mcHeader['n_showers'] * self._numberOfFiles
        )
        self._mcHeader['n_triggered'] = numberOfTriggeredEvents
        return

    @u.quantity_input(coreMax=u.m)
    def countTriggeredEvents(self, energyRange=None, coreMax=None):
        '''
        Count number of triggered events within a certain energy range and core radius.

        Parameters
        ----------
        energyRange: Tuple (len 2)
            Max and min energy of energy range, e.g. energyRange=(100 * u.GeV, 10 * u.TeV)
        coreMax: astropy.Quantity (distance)
            Maximum core radius for selecting showers, e.g. coreMax=1000 * u.m

        Returns
        -------
        int
            Number of triggered events.
        '''
        energyRange = self._validateEnergyRange(energyRange)
        coreMax = self._validateCoreMax(coreMax)

        isInEnergyRange = list(map(
            lambda e: e > energyRange[0] and e < energyRange[1],
            self.summaryEvents['energy']
        ))
        isInCoreRange = list(map(
            lambda r: r < coreMax,
            self.summaryEvents['r_core']
        ))
        return np.sum(np.array(isInEnergyRange) * np.array(isInCoreRange))

    @u.quantity_input(coreMax=u.m)
    def selectEvents(self, energyRange=None, coreMax=None):
        '''
        Select sim_telarray events within a certain energy range and core radius.

        Parameters
        ----------
        energyRange: Tuple (len 2)
            Max and min energy of energy range, e.g. energyRange=(100 * u.GeV, 10 * u.TeV)
        coreMax: astropy.Quantity (distance)
            Maximum core radius for selecting showers, e.g. coreMax=1000 * u.m

        Returns
        -------
        list
            List of events.
        '''
        energyRange = self._validateEnergyRange(energyRange)
        coreMax = self._validateCoreMax(coreMax)

        selectedEvents = list()
        for file in self.inputFiles:
            with SimTelFile(file) as f:

                for event in f:
                    energy = event['mc_shower']['energy']
                    if energy < energyRange[0] or energy > energyRange[1]:
                        continue

                    x_core = event['mc_event']['xcore']
                    y_core = event['mc_event']['ycore']
                    r_core = math.sqrt(math.pow(x_core, 2) + math.pow(y_core, 2))
                    if r_core > coreMax:
                        continue

                    selectedEvents.append(event)
        return selectedEvents

    @u.quantity_input(coreMax=u.m)
    def countSimulatedEvents(self, energyRange=None, coreMax=None):
        '''
        Count (or calculate) number of simulated events within a certain energy range and \
        core radius, nased on the simulated power law.

        Parameters
        ----------
        energyRange: Tuple (len 2)
            Max and min energy of energy range, e.g. energyRange=(100 * u.GeV, 10 * u.TeV)
        coreMax: astropy.Quantity (distance)
            Maximum core radius for selecting showers, e.g. coreMax=1000 * u.m

        Returns
        -------
        int
            Number of simulated events.
        '''
        energyRange = self._validateEnergyRange(energyRange)
        coreMax = self._validateCoreMax(coreMax)

        # energy factor
        def integral(erange):
            power = self._mcHeader['spectral_index'] + 1
            return math.pow(erange[0], power) - math.pow(erange[1], power)

        energy_factor = integral(energyRange) / integral(self._mcHeader['E_range'])

        # core factor
        core_factor = math.pow(coreMax, 2) / math.pow(self._mcHeader['core_range'][1], 2)

        return self._mcHeader['n_events'] * energy_factor * core_factor

    def _validateEnergyRange(self, energyRange):
        '''
        Returns the default energy range from mcHeader in case energyRange=None.
        Checks units, convert it to TeV and return it in the right format, otherwise.
        '''
        if energyRange is None:
            return self._mcHeader['E_range']

        if not isinstance(energyRange[0], u.Quantity) or not isinstance(energyRange[1], u.Quantity):
            msg = 'energyRange must be given as u.Quantity in units of energy'
            self._logger.error(msg)
            raise TypeError(msg)

        try:
            return (energyRange[0].to(u.TeV).value, energyRange[1].to(u.TeV).value)
        except u.core.UnitConversionError:
            msg = 'energyRange must be in units of energy'
            self._logger.error(msg)
            raise TypeError(msg)

    def _validateCoreMax(self, coreMax):
        '''
        Returns the default coreMAx from mcHeader in case coreMax=None.
        Checks units, convert it to m and return it in the right format, otherwise.
        '''
        return self._mcHeader['core_range'][1] if coreMax is None else coreMax.to(u.m).value
