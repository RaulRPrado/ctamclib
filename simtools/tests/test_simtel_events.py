#!/usr/bin/python3

import logging
import unittest

import simtools.io_handler as io
from simtools.simtel.simtel_events import SimtelEvents

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelEvents(unittest.TestCase):

    def setUp(self):
        self.testFiles = list()
        self.testFiles.append(io.getTestDataFile(
            'run201_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst')
        )
        self.testFiles.append(io.getTestDataFile(
            'run202_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst')
        )

    def test_reading_files():
        simtel_events = SimtelEvents(inputFiles=self.testFiles)


if __name__ == '__main__':
    unittest.main()
