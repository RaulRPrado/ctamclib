#!/usr/bin/python3

import logging
from pathlib import Path
import numpy as np
import os
import subprocess
from astropy.io import ascii
from astropy.table import Table
import math
import matplotlib.pyplot as plt

from ctamclib.util import names
from ctamclib.telescope_model import TelescopeModel
from ctamclib.simtel_runner import SimtelRunner
from ctamclib.util.general import collectArguments

__all__ = ['RayTracing']


class RayTracing:
    def __init__(
        self,
        simtelSourcePath,
        telescopeModel,
        label=None,
        filesLocation=None,
        **kwargs
    ):
        ''' Comment '''

        self._simtelSourcePath = simtelSourcePath
        self._filesLocation = Path.cwd() if filesLocation is None else Path(filesLocation)
        self._baseDirectory = self._filesLocation.joinpath('CTAMCFiles')

        self.hasTelescopeModel = False
        self._telescopeModel = None
        self.telescopeModel = telescopeModel

        # Default parameters
        self._zenithAngle = 20                          # deg
        self._offAxisAngle = np.linspace(0.0, 3.0, 7)   # deg
        self._sourceDistance = 10                       # km

        # Label
        self._hasLabel = True
        if label is not None:
            self.label = label
        elif self.hasTelescopeModel:
            self.label = self._telescopeModel.label
        else:
            self._hasLabel = False
            self.label = None

        collectArguments(self, ['zenithAngle', 'offAxisAngle', 'sourceDistance'], **kwargs)
        self._hasResults = False

        # Results file
        fileNameResults = names.rayTracingResultsFileName(
                self._telescopeModel.telescopeType,
                self._sourceDistance,
                self._zenithAngle,
                self.label
        )
        dirResults = self._baseDirectory.joinpath('Results')
        dirResults.mkdir(parents=True, exist_ok=True)
        self._fileResults = dirResults.joinpath(fileNameResults)

        # end of init

    def __repr__(self):
        return 'RayTracing(label={})\n'.format(self.label)

    @property
    def telescopeModel(self):
        return self._telescopeModel

    @telescopeModel.setter
    def telescopeModel(self, tel):
        if isinstance(tel, TelescopeModel):
            self._telescopeModel = tel
            self.hasTelescopeModel = True
        else:
            self._telescopeModel = None
            self.hasTelescopeModel = False
            if tel is not None:
                logging.error('Invalid TelescopeModel')

    def simulate(self, test=False, force=False):
        """ Simulating RayTracing"""
        for thisOffAxis in self._offAxisAngle:
            logging.info('Simulating RayTracing for offAxis={}'.format(thisOffAxis))
            simtel = SimtelRunner(
                simtelSourcePath=self._simtelSourcePath,
                filesLocation=self._filesLocation,
                mode='ray-tracing',
                telescopeModel=self._telescopeModel,
                zenithAngle=self._zenithAngle,
                sourceDistance=self._sourceDistance,
                offAxisAngle=thisOffAxis
            )
            simtel.run(test=test, force=force)

    def analyze(self, export=True, force=False, use_rx=False):
        """ Analyzing RayTracing"""

        if self._fileResults.exists() and not force:
            logging.info('Skipping analyze because file exists and force = False')
            self.readResults()
            return

        focalLength = float(self._telescopeModel.getParameter('focal_length'))
        cmToDeg = 180. / math.pi / focalLength

        self._results = dict()
        self._results['off_axis'] = list()
        self._results['d80_cm'] = list()
        self._results['d80_deg'] = list()
        self._results['eff_area'] = list()
        self._results['eff_flen'] = list()
        self._psfImages = dict()

        for thisOffAxis in self._offAxisAngle:
            logging.info('Analyzing RayTracing for offAxis={}'.format(thisOffAxis))
            photonsFileName = names.rayTracingFileName(
                self._telescopeModel.telescopeType,
                self._sourceDistance,
                self._zenithAngle,
                thisOffAxis,
                self.label,
                'photons'
            )
            file = self._baseDirectory.joinpath('RayTracing').joinpath(photonsFileName)
            image = PSFImage()
            image.readSimtelFile(file)

            if use_rx:
                d80, xPosMean, yPosMean, effArea = self.processRX(file)
                image.d80 = d80
                image.xPosMean = xPosMean
                image.yPosMean = yPosMean
                image.effArea = effArea
            else:
                image.loadPSF()
                d80 = image.d80
                xPosMean = image.xPosMean
                yPosMean = image.yPosMean
                effArea = image.effArea

            self._psfImages[thisOffAxis] = image
            effFlen = (
                'nan' if thisOffAxis == 0 else xPosMean / math.tan(thisOffAxis * math.pi / 180.)
            )
            self._results['off_axis'].append(thisOffAxis)
            self._results['d80_cm'].append(d80)
            self._results['d80_deg'].append(d80 * cmToDeg)
            self._results['eff_area'].append(effArea)
            self._results['eff_flen'].append(effFlen)
        # end for offAxis

        self._hasResults = True

        # Exporting
        if export:
            self.exportResults()

    def processRX(self, file):
        # Use -n to disable the cog optimization
        rxOutput = subprocess.check_output(
            '{}/sim_telarray/bin/rx -f 0.8 -v < {}'.format(self._simtelSourcePath, file),
            shell=True
        )
        rxOutput = rxOutput.split()
        d80 = 2 * float(rxOutput[0])
        xMean = float(rxOutput[1])
        yMean = float(rxOutput[2])
        effArea = float(rxOutput[5])
        return d80, xMean, yMean, effArea

    def exportResults(self):
        logging.info('Exporting results')
        if not self._hasResults:
            logging.error('Cannot export results because it does not exist')
        else:
            table = Table(self._results)
            ascii.write(table, self._fileResults, format='basic', overwrite=True)

    def readResults(self):
        table = ascii.read(self._fileResults, format='basic')
        self._results = dict(table)

    def plot(self, which='d80_cm', **kwargs):
        if which not in ['d80_cm', 'd80_deg', 'eff_area', 'eff_flen']:
            logging.error('Invalid option for plotting RayTracing')
            return

        ax = plt.gca()
        ax.plot(self._results['off_axis'], self._results[which], **kwargs)

    def images(self):
        images = list()
        for thisOffAxis in self._offAxisAngle:
            if thisOffAxis in self._psfImages.keys():
                images.append(self._psfImages[thisOffAxis])
        if len(images) == 0:
            logging.error('No image found')
        return images

# end of RayTracing


class PSFImage:
    def __init__(self):
        """ PSFImage only knows the list of photon position in cm
            No information about the telescope (e.g focal length) should be used in here.
        """
        self.xPos = list()
        self.yPos = list()
        self.xPosMean = None
        self.yPosMean = None
        self.effArea = None
        self.totalPhotons = None
        self.totalArea = None
        self.d80 = None
        self._storedPSF = dict()

    def __repr__(self):
        return 'PSFImage ({}/{} photons)'.format(self.detectedPhotons, self.totalPhotons)

    def readSimtelFile(self, file):
        logging.info('Reading SimtelFile {}'.format(file))
        self.totalPhotons = 0
        self.totalArea = None
        with open(file, 'r') as f:
            for line in f:
                self.processSimtelLine(line)

        if len(self.xPos) == 0 or len(self.yPos) == 0 or len(self.xPos) != len(self.yPos):
            logging.error('Problems reading Simtel file - invalid data')

        self.xPosMean = np.mean(self.xPos)
        self.yPosMean = np.mean(self.yPos)
        self.detectedPhotons = len(self.xPos)
        self.effArea = self.detectedPhotons * self.totalArea / self.totalPhotons

    def loadPSF(self):
        self.d80 = self.getPSF(0.8)

    def processSimtelLine(self, line):
        words = line.split()
        if 'falling on an area of' in line:
            self.totalPhotons += int(words[4])
            area = float(words[14])
            if self.totalArea is None:
                self.totalArea = area
            elif area != self.totalArea:
                logging.error(
                    'Conflicting value of the total area found'
                    ' {} != {}'.format(self.totalArea, area)
                )
        elif '#' in line or len(words) == 0:
            pass
        else:
            self.xPos.append(float(words[2]))
            self.yPos.append(float(words[3]))

    def getPSF(self, fraction):
        if fraction not in self._storedPSF.keys():
            self.computePSF(fraction)
        return self._storedPSF[fraction]

    def computePSF(self, fraction):
        self._storedPSF[fraction] = self.findPSF(fraction)

    def findPSF(self, fraction):
        logging.debug('Finding PSF for fraction = {}'.format(fraction))

        xPos2 = [i**2 for i in self.xPos]
        yPos2 = [i**2 for i in self.yPos]
        xSig = math.sqrt(np.mean(xPos2) - self.xPosMean**2)
        ySig = math.sqrt(np.mean(yPos2) - self.yPosMean**2)
        rSig = math.sqrt(xSig**2 + ySig**2)

        numberTarget = fraction * self.detectedPhotons
        rad = 1.5 * rSig
        number0 = self.sumPhotonsInRadius(rad)
        A = 0.5 * math.sqrt(rad * rad / number0)
        delta = number0 - numberTarget
        nIter = 0
        foundRadius = False
        while not foundRadius and nIter < 100:
            nIter += 1
            dr = -delta * A / math.sqrt(numberTarget)
            while rad + dr < 0:
                dr *= 0.5
            rad += dr
            number = self.sumPhotonsInRadius(rad)
            delta = number - numberTarget
            foundRadius = math.fabs(delta) < self.detectedPhotons / 1000.

        if foundRadius:
            return 2 * rad
        else:
            logging.warning('Could not find PSF efficiently')
            psf = self.findPSFByScanning(numberTarget, rSig)
            return psf

    def findPSFByScanning(self, numberTarget, rSig):
        logging.debug('Finding PSF by scanning')

        def scan(dr, radMin, radMax):
            r0, r1 = radMin, radMin + dr
            s0, s1 = 0, 0
            foundRadius = False
            while not foundRadius:
                s0, s1 = self.sumPhotonsInRadius(r0), self.sumPhotonsInRadius(r1)
                if s0 < numberTarget and s1 > numberTarget:
                    foundRadius = True
                    break
                if r1 > radMax:
                    break
                r0 += dr
                r1 += dr
            if foundRadius:
                return (r0 + r1) / 2, r0, r1
            else:
                logging.error('Could not find PSF by scanning')
                return 0, radMin, radMax

        # Step 0
        rad, radMin, radMax = scan(0.1 * rSig, 0, 4 * rSig)
        # Step 1
        rad, radMin, radMax = scan(0.005 * rSig, radMin, radMax)
        return rad

    def sumPhotonsInRadius(self, radius):
        n = 0
        for x, y in zip(self.xPos, self.yPos):
            d2 = (x - self.xPosMean)**2 + (y - self.yPosMean)**2
            n += 1 if d2 < radius**2 else 0
        return n

    def plot(self, **kwargs):
        ax = plt.gca()
        fac = 1
        xToPlot = fac * np.array(self.xPos) - fac * self.xPosMean
        yToPlot = fac * np.array(self.yPos) - fac * self.yPosMean

        ax.hist2d(xToPlot, yToPlot, bins=80, cmap=plt.cm.gist_heat_r, **kwargs)

        circle = plt.Circle((0, 0), self.d80 / 2, color='black', fill=False, lw=2, ls='--')
        ax.add_artist(circle)

# end of PSFImage
