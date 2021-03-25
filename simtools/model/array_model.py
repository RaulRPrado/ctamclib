import logging
from copy import copy

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.layout.layout_array import LayoutArray
from simtools.util import names
from simtools.util.general import collectDataFromYamlOrDict

__all__ = ['ArrayModel']


class InvalidArrayConfigData(Exception):
    pass


class ArrayModel:
    def __init__(
        self,
        label=None,
        arrayConfigFile=None,
        arrayConfigData=None,
        modelFilesLocations=None,
        filesLocation=None
    ):
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init ArrayModel')

        self.label = label

        self._modelFilesLocations = cfg.getConfigArg('modelFilesLocations', modelFilesLocations)
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)

        arrayConfigData = collectDataFromYamlOrDict(arrayConfigFile, arrayConfigData)
        self._loadArrayData(arrayConfigData)

        self._setConfigFileDirectory()

        self._buildArrayModel()

        # End of init

    def _loadArrayData(self, arrayConfigData):
        ''' Loading parameters from arrayData '''
        # Validating arrayConfigData
        self._validateArrayData(arrayConfigData)

        # Site
        self.site = names.validateSiteName(arrayConfigData['site'])

        # Layout name
        self.layoutName = names.validateLayoutArrayName(arrayConfigData['arrayName'])
        self.layout = LayoutArray.fromLayoutArrayName(
            self.site + '-' + self.layoutName,
            label=self.label
        )

        # Model version
        if 'modelVersion' not in arrayConfigData.keys() or arrayConfigData['modelVersion'] is None:
            self._logger.warning('modelVersion not given in arrayConfigData - using current')
            self.modelVersion = 'current'
        else:
            self.modelVersion = names.validateModelVersionName(arrayConfigData['modelVersion'])

        self._arrayConfigData = {
            k: v for (k, v) in arrayConfigData.items()
            if k not in ['site', 'arrayName', 'modelVersion']
        }

    def _validateArrayData(self, arrayConfigData):
        ''' Validate arrayData by checking the existence of the relevant keys.'''

        def runOverPars(pars, data, parent=None):
            allKeys = data.keys() if parent is None else data[parent].keys()
            for pp in pars:
                if pp not in allKeys:
                    key = pp if parent is None else parent + '.' + pp
                    msg = (
                        'Key {} was not found in arrayConfigData '.format(key)
                        + '- impossible to build array model'
                    )
                    self._logger.error(msg)
                    raise InvalidArrayConfigData(msg)

        runOverPars(['site', 'arrayName', 'default'], arrayConfigData)
        runOverPars(['LST', 'MST'], arrayConfigData, parent='default')
        if names.validateSiteName(arrayConfigData['site']) == 'South':
            runOverPars(['SST'], arrayConfigData, parent='default')
        # End of _validateArrayData

    def _buildArrayModel(self):

        # Getting site parameters from DB
        db = db_handler.DatabaseHandler(self._logger.name)
        self._siteParameters = db.getSiteParameters(
            self.site,
            self.modelVersion,
            self._configFileDirectory,
            onlyApplicable=True
        )

        # Building telescopes
        self._telescopeModel = list()
        self._allTelescopeModelNames = list()
        for tel in self.layout:
            telSize = tel.getTelescopeSize()
            if tel.name in self._arrayConfigData.keys():
                telModelName = self.site + '-' + telSize + '-' + self._arrayConfigData[tel.name]
            else:
                telModelName = (
                    self.site + '-' + telSize + '-' + self._arrayConfigData['default'][telSize]
                )

            if telModelName not in self._allTelescopeModelNames:
                self._allTelescopeModelNames.append(telModelName)
                telModel = TelescopeModel(
                    telescopeName=telModelName,
                    version=self.modelVersion,
                    label=self.label,
                    modelFilesLocations=self._modelFilesLocations,
                    filesLocation=self._filesLocation,
                    logger=self._logger.name
                )
            else:
                for tel in self._telescopeModel:
                    if tel.telescopeName != telModelName:
                        continue
                    self._logger.debug(
                        'Copying tel model {} already loaded from DB'.format(tel.telescopeName)
                    )
                    telModel = copy(tel)

            self._telescopeModel.append(telModel)

        if len(self._telescopeModel) != len(self.layout):
            self._logger.warning(
                'Size of telModel does not match size of layout - something it wrong!'
            )

    def _setConfigFileDirectory(self):
        ''' Define the variable _configFileDirectory and create directories, if needed '''
        self._configFileDirectory = io.getModelOutputDirectory(self._filesLocation, self.label)
        if not self._configFileDirectory.exists():
            self._configFileDirectory.mkdir(parents=True, exist_ok=True)
            self._logger.info('Creating directory {}'.format(self._configFileDirectory))
        return

    def printTelescopeList(self):
        for telData, telModel in zip(self.layout, self._telescopeModel):
            print('Name: {}\t Model: {}'.format(telData.name, telModel.telescopeName))

    # def exportCorsikaInputFile():
    #     pass

    def exportSimtelTelescopeConfigFiles(self):
        '''
        '''
        exportedModels = list()
        for telModel in self._telescopeModel:
            name = telModel.telescopeName
            if name not in exportedModels:
                self._logger.debug('Exporting config file for tel {}'.format(name))
                telModel.exportConfigFile()
                exportedModels.append(name)
            else:
                self._logger.debug('Config file for tel {} already exists - skipping'.format(name))

    def exportSimtelArrayConfigFile(self):
        '''
        '''
        # Setting file name and the location
        configFileName = names.simtelArrayConfigFileName(
            self.layoutName,
            self.site,
            self.modelVersion,
            self.label
        )
        self._configFilePath = self._configFileDirectory.joinpath(configFileName)

        # Writing parameters to the file
        self._logger.info('Writing array config file - {}'.format(self._configFilePath))
        with open(self._configFilePath, 'w') as file:
            header = (
                '%{}\n'.format(50 * '=')
                + '% ARRAY CONFIGURATION FILE\n'
                + '% Site: {}\n'.format(self.site)
                + '% ArrayName: {}\n'.format(self.layoutName)
                + '% ModelVersion: {}\n'.format(self.modelVersion)
                + ('% Label: {}\n'.format(self.label) if self.label is not None else '')
                + '%{}\n\n'.format(50 * '=')
            )
            file.write(header)

            # TELESCOPE 0 - global parameters
            file.write('# if TELESCOPE == 0\n')
            file.write('    echo *****************************\n')
            file.write('    echo Site: {}\n'.format(self.site))
            file.write('    echo ArrayName: {}\n'.format(self.layoutName))
            file.write('    echo ModelVersion: {}\n'.format(self.modelVersion))
            file.write('    echo *****************************\n\n')

            # Writing site parameters
            # HERE

            # Looping over telescopes - from 1 to ...
            for count, telModel in enumerate(self._telescopeModel):
                telConfigFile = telModel.getConfigFile(noExport=True).name
                file.write('# elif TELESCOPE == {} % {}\n\n'.format(
                    count + 1,
                    self.layout[count].name
                ))
                file.write('#  include <{}>\n\n'.format(telConfigFile))
            file.write('# endif \n\n')

    # END exportSimtelArrayConfigFile
