#!/usr/bin/python3
"""Module for visualization."""

import copy
import logging
import re
from collections import OrderedDict
from pathlib import Path

import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from astropy.table import Column, QTable
from cycler import cycler
from matplotlib import gridspec
from matplotlib.collections import PatchCollection

from simtools.utils import geometry as transf
from simtools.utils import names
from simtools.visualization import legend_handlers as leg_h

__all__ = [
    "get_colors",
    "get_lines",
    "get_markers",
    "get_telescope_patch",
    "plot_1d",
    "plot_array",
    "plot_hist_2d",
    "plot_table",
    "save_figure",
    "set_style",
]

COLORS = {}
COLORS["classic"] = [
    "#ba2c54",
    "#5B90DC",
    "#FFAB44",
    "#0C9FB3",
    "#57271B",
    "#3B507D",
    "#794D88",
    "#FD6989",
    "#8A978E",
    "#3B507D",
    "#D8153C",
    "#cc9214",
]
COLORS["modified classic"] = [
    "#D6088F",
    "#424D9C",
    "#178084",
    "#AF99DA",
    "#F58D46",
    "#634B5B",
    "#0C9FB3",
    "#7C438A",
    "#328cd6",
    "#8D0F25",
    "#8A978E",
    "#ffcb3d",
]
COLORS["autumn"] = [
    "#A9434D",
    "#4E615D",
    "#3C8DAB",
    "#A4657A",
    "#424D9C",
    "#DC575A",
    "#1D2D38",
    "#634B5B",
    "#56276D",
    "#577580",
    "#134663",
    "#196096",
]
COLORS["purples"] = [
    "#a57bb7",
    "#343D80",
    "#EA60BF",
    "#B7308E",
    "#E099C3",
    "#7C438A",
    "#AF99DA",
    "#4D428E",
    "#56276D",
    "#CC4B93",
    "#DC4E76",
    "#5C4AE4",
]
COLORS["greens"] = [
    "#268F92",
    "#abc14d",
    "#8A978E",
    "#0C9FB3",
    "#BDA962",
    "#B0CB9E",
    "#769168",
    "#5E93A5",
    "#178084",
    "#B7BBAD",
    "#163317",
    "#76A63F",
]

COLORS["default"] = COLORS["classic"]

MARKERS = ["o", "s", "v", "^", "*", "P", "d", "X", "p", "<", ">", "h"]
LINES = [
    (0, ()),  # solid
    (0, (1, 1)),  # densely dotted
    (0, (3, 1, 1, 1)),  # densely dashdotted
    (0, (5, 5)),  # dashed
    (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
    (0, (5, 1)),  # densely dashed
    (0, (1, 5)),  # dotted
    (0, (3, 5, 1, 5)),  # dashdotted
    (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
    (0, (5, 10)),  # loosely dashed
    (0, (1, 10)),  # loosely dotted
    (0, (3, 10, 1, 10)),  # loosely dashdotted
]

_logger = logging.getLogger(__name__)


def _add_unit(title, array):
    """
    Add a unit to "title" (presumably an axis title).

    The unit is extracted from the unit field of the array, in case array is an astropy quantity.
    If a unit is found, it is added to title in the form [unit]. If a unit already is present in
    title (in the same form), a warning is printed and no unit is added. The function assumes
    array not to be empty and returns the modified title.

    Parameters
    ----------
    title: str
    array: astropy.Quantity

    Returns
    -------
    str
        Title with units.
    """
    unit = ""
    if isinstance(array, u.Quantity):
        unit = str(array[0].unit)
        if len(unit) > 0:
            unit = f" [{unit}]"
        if re.search(r"\d", unit):
            unit = re.sub(r"(\d)", r"^\1", unit)
            unit = unit.replace("[", r"$[").replace("]", r"]$")
        if "[" in title and "]" in title:
            _logger.warning(
                "Tried to add a unit from astropy.unit, "
                "but axis already has an explicit unit. Left axis title as is."
            )
            unit = ""

    return f"{title}{unit}"


def set_style(palette="default", big_plot=False):
    """
    Set the plotting style to homogenize plot style across the framework.

    The function receives the colour palette name and whether it is a big plot or not.\
    The latter sets the fonts and marker to be bigger in case it is a big plot. The available \
    colour palettes are as follows:

    - classic (default): A classic colorful palette with strong colors and contrast.
    - modified classic: Similar to the classic, with slightly different colors.
    - autumn: A slightly darker autumn style colour palette.
    - purples: A pseudo sequential purple colour palette (not great for contrast).
    - greens: A pseudo sequential green colour palette (not great for contrast).

    To use the function, simply call it before plotting anything. The function is made public, so \
    that it can be used outside the visualize module. However, it is highly recommended to create\
    plots only through the visualize module.

    Parameters
    ----------
    palette: str
        Colour palette.
    big_plot: bool
        Flag to set fonts and marker bigger. If True, it sets them bigger.

    Raises
    ------
    KeyError
        if provided palette does not exist.
    """
    if palette not in COLORS:
        raise KeyError(f"palette must be one of {', '.join(COLORS)}")

    fontsize = {"default": 17, "big_plot": 30}
    markersize = {"default": 8, "big_plot": 18}
    plot_size = "big_plot" if big_plot else "default"

    plt.rc("lines", linewidth=2, markersize=markersize[plot_size])
    plt.rc(
        "axes",
        prop_cycle=(
            cycler(color=COLORS[palette]) + cycler(linestyle=LINES) + cycler(marker=MARKERS)
        ),
        titlesize=fontsize[plot_size],
        labelsize=fontsize[plot_size],
        labelpad=5,
        grid=True,
        axisbelow=True,
    )
    plt.rc("xtick", labelsize=fontsize[plot_size])
    plt.rc("ytick", labelsize=fontsize[plot_size])
    plt.rc("legend", loc="best", shadow=False, fontsize="medium")
    plt.rc("font", family="serif", size=fontsize[plot_size])


def get_colors(palette="default"):
    """
    Get the colour list of the palette requested.

    If no palette is provided, the default is returned.

    Parameters
    ----------
    palette: str
        Colour palette.

    Returns
    -------
    list
        Colour list.

    Raises
    ------
    KeyError
        if provided palette does not exist.
    """
    if palette not in COLORS:
        raise KeyError(f"palette must be one of {', '.join(COLORS)}")

    return COLORS[palette]


def get_markers():
    """
    Get the marker list used in this module.

    Returns
    -------
    list
        List with markers.
    """
    return MARKERS


def get_lines():
    """
    Get the line style list used in this module.

    Returns
    -------
    list
        List with line styles.
    """
    return LINES


def filter_plot_kwargs(kwargs):
    """Filter out kwargs that are not valid for plt.plot."""
    valid_keys = {
        "color",
        "linestyle",
        "linewidth",
        "marker",
        "markersize",
        "markerfacecolor",
        "markeredgecolor",
        "markeredgewidth",
        "alpha",
        "label",
        "zorder",
        "dashes",
        "gapcolor",
        "solid_capstyle",
        "solid_joinstyle",
        "dash_capstyle",
        "dash_joinstyle",
        "antialiased",
    }
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def plot_1d(data, **kwargs):
    """
    Produce a high contrast one dimensional plot from multiple data sets.

    A ratio plot can be added at the bottom to allow easy comparison. Additional options,
    such as plot title, plot legend, etc., are given in kwargs. Any option that can be
    changed after plotting (e.g., axes limits, log scale, etc.) should be done using the
    returned plt instance.

    Parameters
    ----------
    data: numpy structured array or a dictionary of structured arrays
          Each structured array has two columns, the first is the x-axis and the second the y-axis.
          The titles of the columns are set as the axes titles.
          The labels of each dataset set are given in the dictionary keys
          and will be used in the legend.
    **kwargs:
        * palette: string
          Choose a colour palette (see set_style for additional information).
        * title: string
          Set a plot title.
        * no_legend: bool
          Do not print a legend for the plot.
        * big_plot: bool
          Increase marker and font sizes (like in a wide light curve).
        * no_markers: bool
          Do not print markers.
        * empty_markers: bool
          Print empty (hollow) markers
        * plot_ratio: bool
          Add a ratio plot at the bottom. The first entry in the data dictionary
          is used as the reference for the ratio.
          If data dictionary is not an OrderedDict, the reference will be random.
        * plot_difference: bool
          Add a difference plot at the bottom. The first entry in the data dictionary
          is used as the reference for the difference.
          If data dictionary is not an OrderedDict, the reference will be random.
        * Any additional kwargs for plt.plot

    Returns
    -------
    pyplot.figure
        Instance of pyplot.figure in which the plot was produced

    Raises
    ------
    ValueError
        if asked to plot a ratio or difference with just one set of data
    """
    kwargs = handle_kwargs(kwargs)
    data_dict, plot_ratio, plot_difference = prepare_data(data, kwargs)
    fig, ax1, gs = setup_plot(kwargs, plot_ratio, plot_difference)
    plot_args = filter_plot_kwargs(kwargs)
    plot_main_data(data_dict, kwargs, plot_args)

    if plot_ratio or plot_difference:
        plot_ratio_difference(ax1, data_dict, plot_ratio, gs, plot_args)

    if not (plot_ratio or plot_difference):
        plt.tight_layout()

    return fig


def handle_kwargs(kwargs):
    """Extract and handle keyword arguments."""
    kwargs_defaults = {
        "palette": "default",
        "big_plot": False,
        "title": "",
        "no_legend": False,
        "no_markers": False,
        "empty_markers": False,
        "plot_ratio": False,
        "plot_difference": False,
        "xscale": "linear",
        "yscale": "linear",
        "xlim": (None, None),
        "ylim": (None, None),
        "xtitle": None,
        "ytitle": None,
    }
    for key, default in kwargs_defaults.items():
        kwargs[key] = kwargs.pop(key, default)

    if kwargs["no_markers"]:
        kwargs["marker"] = "None"
        kwargs["linewidth"] = 4
    if kwargs["empty_markers"]:
        kwargs["markerfacecolor"] = "None"

    return kwargs


def prepare_data(data, kwargs):
    """Prepare data for plotting."""
    if not isinstance(data, dict):
        data_dict = {"_default": data}
    else:
        data_dict = data

    if (kwargs["plot_ratio"] or kwargs["plot_difference"]) and len(data_dict) < 2:
        raise ValueError("Asked to plot a ratio or difference with just one set of data")

    return data_dict, kwargs["plot_ratio"], kwargs["plot_difference"]


def setup_plot(kwargs, plot_ratio, plot_difference):
    """Set up the plot style, figure, and gridspec."""
    set_style(kwargs["palette"], kwargs["big_plot"])

    if plot_ratio or plot_difference:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        fig = plt.figure(figsize=(8, 8))
        gs.update(hspace=0.02)
    else:
        gs = gridspec.GridSpec(1, 1)
        fig = plt.figure(figsize=(8, 6))

    plt.subplot(gs[0])
    ax1 = plt.gca()

    return fig, ax1, gs


def plot_main_data(data_dict, kwargs, plot_args):
    """Plot the main data."""
    for label, data_now in data_dict.items():
        assert len(data_now.dtype.names) == 2, "Input array must have two columns with titles."
        x_column_name, y_column_name = data_now.dtype.names[0], data_now.dtype.names[1]
        x_title = kwargs["xtitle"] if kwargs.get("xtitle") else x_column_name
        y_title = kwargs["ytitle"] if kwargs.get("ytitle") else y_column_name
        x_title_unit = _add_unit(x_title, data_now[x_column_name])
        y_title_unit = _add_unit(y_title, data_now[y_column_name])
        plt.plot(data_now[x_column_name], data_now[y_column_name], label=label, **plot_args)

    plt.xscale(kwargs["xscale"])
    plt.yscale(kwargs["yscale"])
    plt.xlim(kwargs["xlim"])
    plt.ylim(kwargs["ylim"])
    plt.ylabel(y_title_unit)
    if not (kwargs["plot_ratio"] or kwargs["plot_difference"]):
        plt.xlabel(x_title_unit)
    if kwargs["title"]:
        plt.title(kwargs["title"], y=1.02)
    if "_default" not in list(data_dict.keys()) and not kwargs["no_legend"]:
        plt.legend()


def plot_ratio_difference(ax1, data_dict, plot_ratio, gs, plot_args):
    """Plot the ratio or difference plot."""
    plt.subplot(gs[1], sharex=ax1)
    plt.plot([], [])  # Advance cycler for consistent colors/styles

    data_ref_name = next(iter(data_dict))
    for label, data_now in data_dict.items():
        if label == data_ref_name:
            continue
        x_title, y_title = data_now.dtype.names[0], data_now.dtype.names[1]
        x_title_unit = _add_unit(x_title, data_now[x_title])

        if plot_ratio:
            y_values = data_now[y_title] / data_dict[data_ref_name][y_title]
            y_title_ratio = f"Ratio to {data_ref_name}"
        else:
            y_values = data_now[y_title] - data_dict[data_ref_name][y_title]
            y_title_ratio = f"Difference to {data_ref_name}"

        plt.plot(data_now[x_title], y_values, **plot_args)

    plt.xlabel(x_title_unit)

    if len(y_title_ratio) > 20 and plot_ratio:
        y_title_ratio = "Ratio"

    plt.ylabel(y_title_ratio)
    ylim = plt.gca().get_ylim()
    nbins = min(int((ylim[1] - ylim[0]) / 0.05 + 1), 6)
    plt.locator_params(axis="y", nbins=nbins)


def plot_table(table, y_title, **kwargs):
    """
    Produce a high contrast one dimensional plot from the data in an astropy.Table.

    A ratio plot can be added at the bottom to allow easy comparison. Additional options, such
    as plot title, plot legend, etc., are given in kwargs. Any option that can be changed after
    plotting (e.g., axes limits, log scale, etc.) should be done using the returned plt instance.

    Parameters
    ----------
    table: astropy.Table or astropy.QTable
           The first column of the table is the x-axis and the second column is the y-axis. Any \
           additional columns will be treated as additional data to plot. The column titles are \
           used in the legend (except for the first column).
    y_title: str
           The y-axis title.

    **kwargs:
        * palette: choose a colour palette (see set_style for additional information).
        * title: set a plot title.
        * no_legend: do not print a legend for the plot.
        * big_plot: increase marker and font sizes (like in a wide light curve).
        * no_markers: do not print markers.
        * empty_markers: print empty (hollow) markers
        * plot_ratio: bool
          Add a ratio plot at the bottom. The first entry in the data dictionary is used as the \
          reference for the ratio. If data dictionary is not an OrderedDict, the reference will be\
          random.
        * plot_difference: bool
          Add a difference plot at the bottom. The first entry in the data dictionary is used as \
          the reference for the difference. If data dictionary is not an OrderedDict, the reference\
          will be random.
        * Any additional kwargs for plt.plot

    Returns
    -------
    pyplot.fig
        Instance of pyplot.fig.

    Raises
    ------
    ValueError
        if table has less than two columns.
    """
    if len(table.keys()) < 2:
        raise ValueError("Table has to have at least two columns")

    x_axis = table.keys()[0]
    data_dict = OrderedDict()
    for column in table.keys()[1:]:
        data_dict[column] = QTable([table[x_axis], table[column]], names=[x_axis, y_title])

    return plot_1d(data_dict, **kwargs)


def plot_hist_2d(data, **kwargs):
    """
    Produce a two dimensional histogram plot.

    Any option that can be changed after plotting (e.g., axes limits, log scale, etc.)
    should be done using the returned plt instance.

    Parameters
    ----------
    data: numpy structured array
          The columns of the structured array are used as the x-axis and y-axis titles.
    **kwargs:
        * title: set a plot title.
        * Any additional kwargs for plt.hist2d

    Returns
    -------
    pyplot.figure
        Instance of pyplot.figure in which the plot was produced.

    """
    cmap = plt.cm.gist_heat_r
    if "title" in kwargs:
        title = kwargs["title"]
        kwargs.pop("title", None)
    else:
        title = ""

    # Set default style since the usual options do not affect 2d plots (for now).
    set_style()

    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure(figsize=(8, 6))

    ##########################################################################################
    # Plot the data
    ##########################################################################################

    plt.subplot(gs[0])
    assert len(data.dtype.names) == 2, "Input array must have two columns with titles."
    x_title, y_title = data.dtype.names[0], data.dtype.names[1]
    x_title_unit = _add_unit(x_title, data[x_title])
    y_title_unit = _add_unit(y_title, data[y_title])
    plt.hist2d(data[x_title], data[y_title], cmap=cmap, **kwargs)

    plt.xlabel(x_title_unit)
    plt.ylabel(y_title_unit)

    plt.gca().set_aspect("equal", adjustable="datalim")

    if len(title) > 0:
        plt.title(title, y=1.02)

    fig.tight_layout()

    return fig


@u.quantity_input(x=u.m, y=u.m, radius=u.m)
def get_telescope_patch(name, x, y, radius):
    """
    Collect the patch of one telescope to be plotted by plot_array.

    Parameters
    ----------
    name: str
        Name of the telescope (type).
    x: astropy.units.Quantity
        x position of the telescope usually in meters.
    y: astropy.units.Quantity
        y position of the telescope usually in meters.
    radius: astropy.units.Quantity
        Radius of the telescope sphere usually in meters.

    Returns
    -------
    patch
        Instance of mpatches.Circle.
    """
    tel_obj = leg_h.TelescopeHandler()
    valid_name = names.get_array_element_type_from_name(name)
    fill_flag = False

    x = x.to(u.m)
    y = y.to(u.m)
    radius = radius.to(u.m)

    if valid_name.startswith("MST"):
        fill_flag = True
    if valid_name == "SCTS":
        patch = mpatches.Rectangle(
            ((x - radius / 2).value, (y - radius / 2).value),
            width=radius.value,
            height=radius.value,
            fill=False,
            color=tel_obj.colors_dict["SCTS"],
        )
    else:
        patch = mpatches.Circle(
            (x.value, y.value),
            radius=radius.value,
            fill=fill_flag,
            color=tel_obj.colors_dict[valid_name],
        )
    return patch


@u.quantity_input(rotate_angle=u.deg)
def plot_array(
    telescopes, rotate_angle=0, show_tel_label=False, axes_range=None, marker_scaling=1.0
):
    """
    Plot the array of telescopes.

    The x axis gives the easting direction and y axis gives the northing direction.
    Note that in order to convert from the CORSIKA coordinate system to the 'conventional' system
    of North/East, a 90 degree rotation is always applied.
    Rotation of the array elements is possible through the 'rotate_angle' given either in degrees,
    or in radians.
    The direction of rotation of the array elements is counterclockwise.
    The rotation does not change Telescope instance attributes.

    Parameters
    ----------
    telescopes: astropy.table
        Table with the telescope position and names. Note the orientation of the axes.
    rotate_angle:
        Angle to rotate the plot. For rotate_angle = 0 the resulting plot will have
        the x-axis pointing towards the east, and the y-axis pointing towards the North.
    show_tel_label: bool
        If True it will print the label of the individual telescopes in the plot.
        While it works well for the smaller arrays, it gets crowded for larger arrays.
    axes_range : float
        Axis range for both axes. Range is from -plot_range to plot_range.
    maker_scaling : float
        Scaling factor for marker size to be plotted.

    Returns
    -------
    plt.figure
        Instance of plt.figure with the array of telescopes plotted.
    """
    fig, ax = plt.subplots(1)
    legend_objects = []
    legend_labels = []
    tel_counters = initialize_tel_counters()

    pos_x_rotated, pos_y_rotated = get_rotated_positions(telescopes, rotate_angle)
    telescopes.add_column(Column(pos_x_rotated, name="pos_x_rotated"))
    telescopes.add_column(Column(pos_y_rotated, name="pos_y_rotated"))

    fontsize, scale = get_plot_params(len(pos_x_rotated))
    patches = create_patches(
        telescopes, scale, marker_scaling, show_tel_label, ax, fontsize, tel_counters
    )

    update_legend(ax, tel_counters, legend_objects, legend_labels)
    finalize_plot(ax, patches, x_title="Easting [m]", y_title="Northing [m]", axes_range=axes_range)

    return fig


def initialize_tel_counters():
    return {one_telescope: 0 for one_telescope in names.get_list_of_array_element_types()}


def get_rotated_positions(telescopes, rotate_angle):
    pos_x_rotated = pos_y_rotated = None
    if "position_x" in telescopes.colnames and "position_y" in telescopes.colnames:
        pos_x_rotated, pos_y_rotated = telescopes["position_x"], telescopes["position_y"]
        rotate_angle = rotate_angle + 90.0 * u.deg
    elif "utm_east" in telescopes.colnames and "utm_north" in telescopes.colnames:
        pos_x_rotated, pos_y_rotated = telescopes["utm_east"], telescopes["utm_north"]
    else:
        raise ValueError(
            "Telescopes table must contain either 'position_x'/'position_y'"
            "or 'utm_east'/'utm_north' columns"
        )
    if rotate_angle != 0:
        pos_x_rotated, pos_y_rotated = transf.rotate(pos_x_rotated, pos_y_rotated, rotate_angle)
    return pos_x_rotated, pos_y_rotated


def get_plot_params(position_length):
    if position_length > 30:
        return 4, 2
    return 8, 1


def create_patches(telescopes, scale, marker_scaling, show_tel_label, ax, fontsize, tel_counters):
    patches = []
    for tel_now in telescopes:
        telescope_name = get_telescope_name(tel_now)
        update_tel_counters(tel_counters, telescope_name)
        sphere_radius = get_sphere_radius(tel_now)
        i_tel_name = names.get_array_element_type_from_name(telescope_name)
        patches.append(
            get_telescope_patch(
                i_tel_name,
                tel_now["pos_x_rotated"],
                tel_now["pos_y_rotated"],
                scale * sphere_radius * marker_scaling,
            )
        )
        if show_tel_label:
            ax.text(
                tel_now["pos_x_rotated"].value,
                tel_now["pos_y_rotated"].value + scale * sphere_radius.value,
                telescope_name,
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=fontsize,
            )
    return patches


def get_telescope_name(tel_now):
    """Get the telescope name from the table row."""
    try:
        return tel_now["telescope_name"]
    except KeyError:
        return tel_now["asset_code"] + "-" + tel_now["sequence_number"]


def update_tel_counters(tel_counters, telescope_name):
    """Update the counter for the given telescope type."""
    for tel_type in tel_counters:
        if tel_type in telescope_name:
            tel_counters[tel_type] += 1


def get_sphere_radius(tel_now):
    """Get the sphere radius of the telescope."""
    return 1.0 * u.m if "sphere_radius" not in tel_now.colnames else tel_now["sphere_radius"]


def update_legend(ax, tel_counters, legend_objects, legend_labels):
    """Update the legend with the telescope counts."""
    for one_telescope in names.get_list_of_array_element_types():
        if tel_counters[one_telescope] > 0:
            legend_objects.append(leg_h.all_telescope_objects[one_telescope]())
            legend_labels.append(f"{one_telescope} ({tel_counters[one_telescope]})")
    legend_handler_map = {k: v() for k, v in leg_h.legend_handler_map.items()}
    ax.legend(
        legend_objects,
        legend_labels,
        handler_map=legend_handler_map,
        prop={"size": 11},
        loc="best",
    )


def finalize_plot(ax, patches, x_title, y_title, axes_range):
    """Finalize the plot by adding titles, setting limits, and adding patches."""
    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.set_xlabel(x_title, fontsize=12, labelpad=0)
    ax.set_ylabel(y_title, fontsize=12, labelpad=0)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_axisbelow(True)
    ax.axis("square")
    if axes_range is not None:
        ax.set_xlim(-axes_range, axes_range)
        ax.set_ylim(-axes_range, axes_range)
    plt.tight_layout()


def plot_simtel_ctapipe(filename, cleaning_args, distance, return_cleaned=False):
    """
    Read in a simtel file and plots reconstructed photoelectrons via ctapipe.

    Parameters
    ----------
    filename : str
        Path to the simtel file.
    cleaning_args : tuple, optional
        Cleaning parameters as (boundary_thresh, picture_thresh, min_number_picture_neighbors).
    distance : astropy Quantity, optional
        Distance to the target.
    return_cleaned : bool, optional
        If True, apply cleaning to the image.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure containing the plot.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.calib import CameraCalibrator
    from ctapipe.image import tailcuts_clean
    from ctapipe.io import EventSource
    from ctapipe.visualization import CameraDisplay

    default_cleaning_levels = {
        "CHEC": (2, 4, 2),
        "LSTCam": (3.5, 7, 2),
        "FlashCam": (3.5, 7, 2),
        "NectarCam": (4, 8, 2),
    }

    source = EventSource(filename, max_events=1)
    event = None
    events = [copy.deepcopy(event) for event in source]
    if len(events) > 1:
        event = events[-1]
    else:
        try:
            event = events[0]
        except IndexError:
            event = events
    tel_id = sorted(event.r1.tel.keys())[0]

    calib = CameraCalibrator(subarray=source.subarray)
    calib(event)

    geometry = source.subarray.tel[1].camera.geometry
    image = event.dl1.tel[tel_id].image
    cleaned = image.copy()

    if return_cleaned:
        if cleaning_args is None:
            boundary, picture, min_neighbors = default_cleaning_levels[geometry.name]
        else:
            boundary, picture, min_neighbors = cleaning_args
        mask = tailcuts_clean(
            geometry,
            image,
            picture_thresh=picture,
            boundary_thresh=boundary,
            min_number_picture_neighbors=min_neighbors,
        )
        cleaned[~mask] = 0

    fig, ax = plt.subplots(dpi=300)
    title = f"CT{tel_id}, run {event.index.obs_id} event {event.index.event_id}"
    disp = CameraDisplay(geometry, image=cleaned, norm="symlog", ax=ax)
    disp.cmap = "RdBu_r"
    disp.add_colorbar(fraction=0.02, pad=-0.1)
    disp.set_limits_percent(100)
    ax.set_title(title, pad=20)
    ax.annotate(
        f"tel type: {source.subarray.tel[1].type.name}\n"
        f"optics: {source.subarray.tel[1].optics.name}\n"
        f"camera: {source.subarray.tel[1].camera_name}\n"
        f"distance: {distance.to(u.m)}",
        xy=(0, 0),
        xytext=(0.1, 1),
        xycoords="axes fraction",
        va="top",
        size=7,
    )
    ax.annotate(
        f"dl1 image,\ntotal $p.e._{{reco}}$: {np.round(np.sum(image))}\n",
        xy=(0, 0),
        xytext=(0.75, 1),
        xycoords="axes fraction",
        va="top",
        ha="left",
        size=7,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def save_figure(fig, output_file, figure_format=None, log_title=""):
    """
    Save figure to output file(s).

    Parameters
    ----------
    fig: plt.figure
        Figure to save.
    output_file: Path, str
        Path to save the figure (without suffix).
    figure_format: list
        List of formats to save the figure.
    title: str
        Title of the figure to be added to the log message.
    """
    figure_format = figure_format or ["pdf", "png"]
    for fmt in figure_format:
        _file = Path(output_file).with_suffix(f".{fmt}")
        fig.savefig(_file, format=fmt, bbox_inches="tight")
        logging.info(f"Saved plot {log_title} to {_file}")

    fig.clf()
