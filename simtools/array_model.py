#!/usr/bin/python3

""" This module contains the ArrayModel class

    Todo:

"""

import logging
import yaml
from pathlib import Path

from simtools.util import names
from simtools import db_handler as db

__all__ = ['getArray', 'ArrayModel']


def getArray(arrayName, databaseLocation):
    """ Return the telescope size (SST, MST or LST) for a given telescopeType.

    Args:
        arrayName (str):

        databaseLocation (Path):

    Returns:
        dict:

    """

    arrayName = names.validateName(arrayName, names.allArrayNames)
    allArrays = db.getArrayDB(databaseLocation)
    return allArrays[arrayName]


class ArrayModel:
    """MC Model handler at the telescope level.

    TelescopeModel is an abstract representation of the MC model
    at the telescope level. It contains the list of parameters and
    useful methods to handle it.

    Attributes:
        yamlDBPath (str): path of the yaml database containing the model parameters.
        label (str): instance label to avoid conflict between files.
        version (str): MC model version (ex. prod4).
        telescopeType (str): telescope type for the base set of parameters (ex. SST-2M-ASTRI,
            LST, ...)
        site (str): Paranal(south)/LaPalma(north).
        filesLocation (str): location for created files (ex. simtel cfg file). If None,
            pwd will used.

    """
    def __init__(
        self,
        telescopeType,
        site,
        yamlDBPath=None,
        version='default',
        label=None,
        filesLocation=None,
        readFromDB=True
    ):
        """TelescopeModel __init__.

        Args:
            telescopeType (str): telescope type for the base set of parameters (ex. SST-2M-ASTRI,
                LST, ...)
            site (str): Paranal(south)/LaPalma(north).
            yamlDBPath (str): path of the yaml database containing the model parameters.
            version (str): MC model version (ex. prod4).
            label (str): instance label to avoid conflict between files.
            filesLocation (str): location for created files (ex. simtel cfg file). If None,
                pwd will used.
            readFromDB (bool): if True, parameters will be read from DB at the initizalition.
                It must be True most of the cases. It must be False specially if classmethod
                fromConfigFile is used.

        """
        self.log = logging.getLogger(__name__)
        self.log.info('Init TelescopeModel')

        self.yamlDBPath = yamlDBPath
        self.label = label
        self._version = None
        self.version = version
        self._telescopeType = None
        self.telescopeType = telescopeType
        self._site = None
        self.site = site
        self._parameters = dict()

        self._filesLocation = Path.cwd() if filesLocation is None else Path(filesLocation)

        if readFromDB:
            self._loadParametersFromDB()

        self._isFileExported = False

    @property
    def version(self):
        """ str: model version after validation. """
        return self._version

    @version.setter
    def version(self, value):
        self._version = names.validateName(value, names.allModelVersionNames)

    @property
    def telescopeType(self):
        """str: telescope type after validation. """
        return self._telescopeType

    @telescopeType.setter
    def telescopeType(self, value):
        self._telescopeType = names.validateName(value, names.allTelescopeTypeNames)

    @property
    def site(self):
        """str: site after validation. """
        return self._site

    @site.setter
    def site(self, value):
        self._site = names.validateName(value, names.allSiteNames)

    @classmethod
    def fromConfigFile(cls, configFileName, telescopeType, site, label=None, filesLocation=None):
        """ Create a TelescopeModel from a sim_telarray config file.

            Todo:
                - dealing with ifdef/indef etc. By now it just keeps the last version of the
                    parameters in the file.

            Args:
                telescopeType (str):
                site (str):
                label (str):
                configFileName (str): path to the input config file

            Return:
                Instance of TelescopeModel class

        """
        parameters = dict()
        tel = cls(
            telescopeType=telescopeType,
            site=site,
            label=label,
            filesLocation=filesLocation,
            readFromDB=False
        )

        def processLine(words):
            iComment = len(words)
            for w in words:
                if '%' in w:
                    iComment = words.index(w)
                    break
            words = words[0:iComment]
            par = words[0].replace('=', '')
            value = ''
            for w in words[1:]:
                w = w.replace('=', '')
                w = w.replace(',', ' ')
                value += w + ' '
            return par, value

        with open(configFileName, 'r') as file:
            for line in file:
                words = line.split()
                if len(words) == 0:
                    continue
                elif '%' in words[0] or 'echo' in words:
                    continue
                elif '#' not in line and len(words) > 0:
                    par, value = processLine(words)
                    par, value = tel._validateParameter(par, value)
                    parameters[par] = value

        tel.addParameters(**parameters)
        return tel

    def _loadParametersFromDB(self):
        """ Read parameters from DB and store it in _parameters (dict). """

        def _readParsFromOneType(yamlDBPath, telescopeType, parametersDB):
            """ Read parameters as a dict and concatenate it to parametersDB
                Implementation was needed to concatenate the optics parameters
                to the MST models.
            """
            fileNameDB = '{}/parValues-{}.yml'.format(yamlDBPath, telescopeType)
            self.log.info('Reading DB file {}'.format(fileNameDB))
            with open(fileNameDB, 'r') as stream:
                pars = yaml.load(stream, Loader=yaml.FullLoader)
            parametersDB.update(pars)

        parametersDB = dict()
        _readParsFromOneType(self.yamlDBPath, self.telescopeType, parametersDB)
        if whichTelescopeSize(self.telescopeType) == 'MST':
            _readParsFromOneType(self.yamlDBPath, 'MST-optics', parametersDB)

        for parNameIn in parametersDB:
            if parametersDB[parNameIn]['Applicable']:
                parName, parValue = self._validateParameter(
                    parNameIn,
                    parametersDB[parNameIn][self._version]
                )
                self._parameters[parName] = parValue

        # Site
        # Two site parameters need to be read:
        # atmospheric_transmission and altitude
        self.log.debug('Reading site parameters from DB')

        def _getSiteParameter(yamlDBPath, site, parName):
            """ Get the value of parName for a given site """
            fileNameDB = '{}/parValues-Sites.yml'.format(yamlDBPath)
            self.log.info('Reading DB file {}'.format(fileNameDB))
            with open(fileNameDB, 'r') as stream:
                allPars = yaml.load(stream, Loader=yaml.FullLoader)
                for par in allPars:
                    if parName in par and self.site.lower() in par:
                        return allPars[par][self._version]

        for siteParName in ['atmospheric_transmission', 'altitude']:
            siteParValue = _getSiteParameter(self.yamlDBPath, self.site, siteParName)
            parName, parValue = self._validateParameter(siteParName, siteParValue)
            self._parameters[parName] = parValue

    # end _loadParametersFromDB

    def _validateParameter(self, parNameIn, parValueIn):
        """ Validate model parameter based on the dict MODEL_PARS.

            Args:
                parNameIn (str): name of the parameters
                parValueIn (str): value of parameter

            Return:
                parNameIn, parValueIn after validated. parValueIn is converted to the proper
                type if that information is available in MODEL_PARS
        """
        self.log.debug('Validating parameter {}'.format(parNameIn))
        for parNameModel in MODEL_PARS.keys():
            if parNameIn == parNameModel or parNameIn in MODEL_PARS[parNameModel]['names']:
                parType = MODEL_PARS[parNameModel]['type']
                return parNameModel, parType(parValueIn)
        return parNameIn, parValueIn

    def getParameter(self, parName):
        """ Get an EXISTING parameter of the model.

            Args:
                parName (str): name of the parameters

            Return:
                Value of the parameter

            Raises:
                ValueError: if parName does not match any parameter in the model.
        """

        if parName in self._parameters:
            return self._parameters[parName]
        else:
            raise ValueError('Parameter {} was not found in the model'.format(parName))

    def addParameters(self, **kwargs):
        """ Add a NEW parameters to the model.

            Args:
                kwargs: parameters should be passed as parameterName=value.

            Raises:
                ValueError: if an existing parameter is tried to be set added.
        """
        for par in kwargs.keys():
            if par in self._parameters.keys():
                raise ValueError(
                    'Parameter {} already in the model, use changeParameter instead'.format(par)
                )
            else:
                self.log.info('Adding {}={} to the model'.format(par, kwargs[par]))
                self._parameters[par] = str(kwargs[par])

    def changeParameters(self, **kwargs):
        """ Change the value of EXISTING parameters to the model.

            Args:
                kwargs: parameters should be passed as parameterName=value.

            Raises:
                ValueError: if the parameter to be changed does not exist.
        """
        for par in kwargs.keys():
            if par not in self._parameters.keys():
                raise ValueError(
                    'Parameter {} not in the model, use addParameters instead'.format(par)
                )
            else:
                self._parameters[par] = kwargs[par]

    def removeParameters(self, *args):
        """ Remove a parameter from the model.

            Args:
                args: each parameter to be removed has to be passed as args.

            Raises:
                ValueError: if the parameter to be removed is not on the model.
        """
        for par in args:
            if par in self._parameters.keys():
                self.log.info('Removing parameter {}'.format(par))
                del self._parameters[par]
            else:
                raise ValueError(
                    'Could not remove parameter {} because it does not exist'.format(par)
                )

    def exportConfigFile(self):
        """ Export the config file used by sim_telarray. """

        # Setting file name and the location
        configFileName = 'CTA-{}-{}-{}'.format(self._version, self.site, self.telescopeType)
        configFileName += '_{}'.format(self.label) if self.label is not None else ''
        configFileName += '.cfg'

        configFileDirectory = io.getModelOutputDirectory(self._filesLocation, self.label)

        if not configFileDirectory.exists():
            configFileDirectory.mkdir(parents=True, exist_ok=True)
            self.log.info('Creating directory {}'.format(configFileDirectory))
        self._configFilePath = configFileDirectory.joinpath(configFileName)

        # Writing parameters to the file
        self.log.info('Writing config file - {}'.format(self._configFilePath))
        with open(self._configFilePath, 'w') as file:
            header = ('%{}\n'.format(99 * '=')
                      + '% Configuration file for:\n'
                      + '% TelescopeType: {}\n'.format(self._telescopeType)
                      + '% Site: {}\n'.format(self.site)
                      + '% Label: {}\n'.format(self.label)
                      + '%{}\n'.format(99 * '='))

            file.write(header)
            for par in self._parameters:
                value = self._parameters[par]
                file.write('{} = {}\n'.format(par, value))

        self._isFileExported = True
    # end exportConfigFile

    def getConfigFile(self):
        """ Return the full path of the config file for sim_telarray.
            The config file is produced if the file still does not exist.

            Return:
                Path of the config file for sim_telarray.
        """
        if not self._isFileExported:
            self.exportConfigFile()
        return self._configFilePath
