#!/usr/bin/python3

import pytest
from astropy.table import Table

from simtools.io_operations import legacy_data_handler


@pytest.fixture
def test_spe_file():
    return "tests/resources/SinglePhe_spectrum_totalfit_19pixel-average_20200601.csv"


def test_read_legacy_data_file(test_spe_file):

    table = legacy_data_handler.read_legacy_data_as_table(test_spe_file, "legacy_lst_single_pe")
    assert table.colnames == ["amplitude", "response"]

    with pytest.raises(ValueError, match="Unsupported legacy data file type: not_a_file_type"):
        legacy_data_handler.read_legacy_data_as_table(test_spe_file, "not_a_file_type")


def test_read_legacy_lst_single_pe(test_spe_file):

    assert isinstance(legacy_data_handler.read_legacy_lst_single_pe(test_spe_file), Table)
