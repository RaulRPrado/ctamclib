"""Calculate the thresholds for energy, radial distance, and viewcone."""

import astropy.units as u
import numpy as np


class LimitCalculator:
    """
    Compute thresholds/limits for energy, radial distance, and viewcone.

    Histograms are generated with simtools-generate-simtel-array-histograms with --hdf5 flag.

    Event data is read from the generated HDF5 file from the following tables:
    - angle_to_observing_position__triggered_showers_ for the viewcone limit.
    - event_weight__ra3d__log10_e__ for the energy and radial distance limit.


    Parameters
    ----------
    event_data_file : list of astropy.table.Table
        The list of tables containing the event data.
    """

    def __init__(self, event_data_file_tables):
        """
        Initialize the LimitCalculator with the given event data file.

        Parameters
        ----------
        event_data_file : list of astropy.table.Table
            The list of tables containing the event data.
        """
        self.angle_to_observing_position__triggered_showers_ = None
        self.event_weight__ra3d__log10_e__ = None

        for table in event_data_file_tables:
            if (
                "Title" in table.meta
                and table.meta["Title"] == "angle_to_observing_position__triggered_showers_"
            ):
                self.angle_to_observing_position__triggered_showers_ = table
            elif "Title" in table.meta and table.meta["Title"] == "event_weight__ra3d__log10_e__":
                self.event_weight__ra3d__log10_e__ = table

    def _compute_limits(
        self, event_weight_array, bin_edges, loss_fraction, axis=0, limit_type="lower"
    ):
        """
        Compute the limits based on the loss fraction.

        Parameters
        ----------
        event_weight_array : np.ndarray
            Array of event weights.
        bin_edges : np.ndarray
            Array of bin edges.
        loss_fraction : float
            Fraction of events to be lost.
        axis : int, optional
            Axis along which to sum the event weights. Default is 0.
        limit_type : str, optional
            Type of limit ('lower' or 'upper'). Default is 'lower'.

        Returns
        -------
        int
            Bin index where the threshold is reached.
        float
            Bin edge value corresponding to the threshold.
        """
        projection = np.sum(event_weight_array, axis=axis)
        bin_edge_value = None
        cumulative_sum = None
        if limit_type == "upper":
            cumulative_sum = np.cumsum(projection)

        elif limit_type == "lower":
            cumulative_sum = np.cumsum(projection[::-1])

        total_events = np.sum(projection)
        threshold = (1 - loss_fraction) * total_events
        bin_index = np.searchsorted(cumulative_sum, threshold)
        if limit_type == "upper":
            bin_edge_value = bin_edges[bin_index]
        elif limit_type == "lower":
            bin_edge_value = bin_edges[-bin_index]
        return bin_index, bin_edge_value

    def get_bin_edges_and_units(self, table, axis="x"):
        """
        Extract bin edges and units from the table metadata.

        Parameters
        ----------
        table : astropy.table.Table
            Table containing the event data.

        Returns
        -------
        tuple
            Tuple containing the bin edges and their units.
        """
        bin_edges = table.meta[f"{axis}_bin_edges"]
        try:
            bin_edges_unit = table.meta[f"{axis}_bin_edges_unit"]
        except KeyError:
            bin_edges_unit = ""
        return bin_edges, bin_edges_unit

    def compute_lower_energy_limit(self, loss_fraction):
        """
        Compute the lower energy limit in TeV based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        astropy.units.Quantity
            Lower energy limit.
        """
        event_weight_array = np.column_stack(
            [
                self.event_weight__ra3d__log10_e__[name]
                for name in self.event_weight__ra3d__log10_e__.dtype.names
            ]
        )
        bin_edges, bin_edges_unit = self.get_bin_edges_and_units(
            self.event_weight__ra3d__log10_e__, axis="y"
        )
        if bin_edges_unit == "":
            bin_edges_unit = "TeV"
        _, lower_bin_edge_value = self._compute_limits(
            event_weight_array, bin_edges, loss_fraction, axis=0, limit_type="lower"
        )
        return (10**lower_bin_edge_value) * u.Unit(bin_edges_unit)

    def compute_upper_radial_distance(self, loss_fraction):
        """
        Compute the upper radial distance based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        astropy.units.Quantity
            Upper radial distance in m.
        """
        event_weight_array = np.column_stack(
            [
                self.event_weight__ra3d__log10_e__[name]
                for name in self.event_weight__ra3d__log10_e__.dtype.names
            ]
        )
        bin_edges, bin_edges_unit = self.get_bin_edges_and_units(
            self.event_weight__ra3d__log10_e__, axis="x"
        )
        if bin_edges_unit == "":
            bin_edges_unit = "m"
        _, upper_bin_edge_value = self._compute_limits(
            event_weight_array, bin_edges, loss_fraction, axis=1, limit_type="upper"
        )
        return upper_bin_edge_value * u.Unit(bin_edges_unit)

    def compute_viewcone(self, loss_fraction):
        """
        Compute the viewcone based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        astropy.units.Quantity
            Viewcone radius in degrees.
        """
        angle_to_observing_position__triggered_showers = np.column_stack(
            [
                self.angle_to_observing_position__triggered_showers_[name]
                for name in self.angle_to_observing_position__triggered_showers_.dtype.names
            ]
        )
        bin_edges, bin_edges_unit = self.get_bin_edges_and_units(
            self.angle_to_observing_position__triggered_showers_, axis="x"
        )
        if bin_edges_unit == "":
            bin_edges_unit = "deg"
        _, upper_bin_edge_value = self._compute_limits(
            angle_to_observing_position__triggered_showers,
            bin_edges,
            loss_fraction,
            axis=0,
            limit_type="upper",
        )
        return upper_bin_edge_value * u.Unit(bin_edges_unit)
