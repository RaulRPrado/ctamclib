#!/usr/bin/python3

import logging


__all__ = ['SimtelConfigWriter']


class SimtelConfigWriter:
    '''
    '''
    TAB = ' ' * 3
    SITE_PARS_TO_WRITE_IN_CONFIG = ['altitude', 'atmospheric_transmission']
    COMMON_PARS = {
        'trigger_telescopes': 1,
        'array_trigger': 'none',
        'trigger_telescopes': 2,
        'only_triggered_telescopes': 1,
        'array_window': 1000,
        'output_format': 1,
        'mirror_list': 'none',
        'telescope_random_angle': 0.,
        'telescope_random_error': 0.,
        'convergent_depth': 0,
        'iobuf_maximum': 1000000000,
        'iobuf_output_maximum': 400000000,
        'multiplicity_offset': -0.5,
        'discriminator_pulse_shape': 'none',
        'discriminator_amplitude': 0.,
        'discriminator_threshold': 99999.,
        'fadc_noise': 0.,
        'asum_threshold': 0.,
        'asum_shaping_file': 'none',
        'asum_offset': 0.0,
        'dsum_threshold': 0,
        'fadc_pulse_shape': 'none',
        'fadc_amplitude': 0.,
        'fadc_pedestal': 100.,
        'fadc_max_signal': 4095,
        'fadc_max_sum': 16777215,
        'store_photoelectrons': 30,
        'pulse_analysis': -30,
        'sum_before_peak': 3,
        'sum_after_peak': 4
    }

    def __init__(self, site, modelVersion, layoutName=None, telescopeName=None, label=None):
        '''
        '''

        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init SimtelConfigWriter')

        self._site = site
        self._modelVersion = modelVersion
        self._label = label
        self._layoutName = layoutName
        self._telescopeName = telescopeName

    def writeSimtelTelescopeConfigFile(self, configFilePath, parameters):
        '''
        '''
        with open(configFilePath, 'w') as file:
            self._writeHeader(file, 'TELESCOPE CONFIGURATION FILE')

            file.write('#ifdef TELESCOPE\n')
            file.write(
                '   echo Configuration for {}'.format(self._telescopeName)
                + ' - TELESCOPE $(TELESCOPE)\n'
            )
            file.write('#endif\n\n')

            for par in parameters.keys():
                value = parameters[par]['Value']
                file.write('{} = {}\n'.format(par, value))

    def writeSimtelArrayConfigFile(
        self,
        configFilePath,
        layout,
        telescopeModel,
        siteParameters
    ):
        '''
        '''
        with open(configFilePath, 'w') as file:
            self._writeHeader(file, 'ARRAY CONFIGURATION FILE')

            # Be carefull with the formating - simtel is sensitive
            file.write('#ifndef TELESCOPE\n')
            file.write('# define TELESCOPE 0\n')
            file.write('#endif\n\n')

            # TELESCOPE 0 - global parameters
            file.write('#if TELESCOPE == 0\n')
            file.write(self.TAB + 'echo *****************************\n')
            file.write(self.TAB + 'echo Site: {}\n'.format(self._site))
            file.write(self.TAB + 'echo LayoutName: {}\n'.format(self._layoutName))
            file.write(self.TAB + 'echo ModelVersion: {}\n'.format(self._modelVersion))
            file.write(self.TAB + 'echo *****************************\n\n')

            # Writing site parameters
            self._writeSiteParameters(file, siteParameters)

            # Writing common parameters
            self._writeCommonParameters(file)

            # Maximum telescopes
            file.write(self.TAB + 'maximum_telescopes = {}\n\n'.format(len(telescopeModel)))

            # Default telescope - 0th tel in telescope list
            telConfigFile = (telescopeModel[0].getConfigFile(noExport=True).name)
            file.write('# include <{}>\n\n'.format(telConfigFile))

            # Looping over telescopes - from 1 to ...
            for count, telModel in enumerate(telescopeModel):
                telConfigFile = telModel.getConfigFile(noExport=True).name
                file.write('%{}\n'.format(layout[count].name))
                file.write('#elif TELESCOPE == {}\n\n'.format(count + 1))
                file.write('# include <{}>\n\n'.format(telConfigFile))
            file.write('#endif \n\n')
    # END writeSimtelArrayConfigFile

    def _writeHeader(self, file, title):
        header = '%{}\n'.format(50 * '=')
        header += '% {}\n'.format(title)
        header += '% Site: {}\n'.format(self._site)
        header += '% ModelVersion: {}\n'.format(self._modelVersion)
        header += (
            '% TelescopeName: {}\n'.format(self._telescopeName)
            if self._telescopeName is not None else ''
        )
        header += (
            '% LayoutName: {}\n'.format(self._layoutName) if self._layoutName is not None else ''
        )
        header += ('% Label: {}\n'.format(self._label) if self._label is not None else '')
        header += '%{}\n\n'.format(50 * '=')
        file.write(header)

    def _writeSiteParameters(self, file, siteParameters):
        file.write(self.TAB + '% Site parameters\n')
        for par in siteParameters:
            if par not in self.SITE_PARS_TO_WRITE_IN_CONFIG:
                continue
            value = siteParameters[par]['Value']
            file.write(self.TAB + '{} = {}\n'.format(par, value))
        file.write('\n')

    def _writeCommonParameters(self, file):
        # Common parameters taken from CTA-PROD4-common.cfg
        # TODO: Store these somewhere else
        self._logger.warning('Common parameters are hardcoded!')

        for par, value in self.COMMON_PARS.items():
            file.write('   {} = {}\n'.format(par, value))
    # End of writeCommonParameters
