#!/usr/bin/python3

import numpy as np

from simtools.io_operations.hdf5_handler import fill_hdf5_table, read_hdf5


def test_fill_hdf5_table_1d(corsika_histograms_instance_set_histograms):
    hist = np.array([1, 2, 3])
    x_bin_edges = np.array([1, 2, 3, 4])
    y_bin_edges = None
    x_label = "test_x_label"
    y_label = None

    table = fill_hdf5_table(
        hist,
        x_bin_edges,
        y_bin_edges,
        x_label,
        y_label,
        corsika_histograms_instance_set_histograms._meta_dict,
    )

    assert all(table[x_label] == x_bin_edges[:-1])
    assert all(table["values"] == hist)


def test_fill_hdf5_table_2d(corsika_histograms_instance_set_histograms):
    hist = np.array([[1, 2], [3, 4]])
    x_bin_edges = np.array([1, 2, 3])
    y_bin_edges = np.array([1, 2, 3])
    x_label = "test_x_label"
    y_label = "test_y_label"

    table = fill_hdf5_table(
        hist,
        x_bin_edges,
        y_bin_edges,
        x_label,
        y_label,
        corsika_histograms_instance_set_histograms._meta_dict,
    )
    assert all(table["test_y_label_0"] == np.array([1, 2]))
    assert all(table["test_y_label_1"] == np.array([3, 4]))


def test_read_hdf5():

    tables = read_hdf5(
        "tests/resources/run2_gamma_za20deg_azm0deg-North-Prod5_test-production-5_reduced.hdata.hdf5"
    )
    assert len(tables) == 4
