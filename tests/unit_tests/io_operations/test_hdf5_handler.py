#!/usr/bin/python3

import numpy as np
import pytest

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

    assert all(table.meta["x_bin_edges"] == x_bin_edges)
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
    assert all(table.meta["x_bin_edges"] == x_bin_edges)
    assert all(table.meta["y_bin_edges"] == y_bin_edges)


def test_read_hdf5():

    tables = read_hdf5(
        "tests/resources/run2_gamma_za20deg_azm0deg-North-Prod5_test-production-5_reduced.hdata.hdf5"
    )
    assert len(tables) == 4


def test_fill_hdf5_table_wrong_dimensions(corsika_histograms_instance_set_histograms):
    hist = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    x_bin_edges = np.array([1, 2, 3, 4])
    y_bin_edges = None
    x_label = "test_x_label"
    y_label = None

    with pytest.raises(ValueError, match="Histogram must be either 1D or 2D."):
        fill_hdf5_table(
            hist,
            x_bin_edges,
            y_bin_edges,
            x_label,
            y_label,
            corsika_histograms_instance_set_histograms._meta_dict,
        )


def test_fill_hdf5_table_1d_with_y_bin_edges(corsika_histograms_instance_set_histograms):
    hist = np.array([1, 2, 3])
    x_bin_edges = np.array([1, 2, 3, 4])
    y_bin_edges = np.array([1, 2, 3])
    x_label = "test_x_label"
    y_label = None

    with pytest.raises(ValueError, match="y_bin_edges should be None for 1D histograms."):
        fill_hdf5_table(
            hist,
            x_bin_edges,
            y_bin_edges,
            x_label,
            y_label,
            corsika_histograms_instance_set_histograms._meta_dict,
        )


def test_fill_hdf5_table_2d_without_y_bin_edges(corsika_histograms_instance_set_histograms):
    hist = np.array([[1, 2], [3, 4]])
    x_bin_edges = np.array([1, 2, 3])
    y_bin_edges = None
    x_label = "test_x_label"
    y_label = "test_y_label"

    with pytest.raises(ValueError, match="y_bin_edges should not be None for 2D histograms."):
        fill_hdf5_table(
            hist,
            x_bin_edges,
            y_bin_edges,
            x_label,
            y_label,
            corsika_histograms_instance_set_histograms._meta_dict,
        )
