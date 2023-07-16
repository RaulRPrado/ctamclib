import logging

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

_logger = logging.getLogger(__name__)


def _kernel_plot_2D_photons(corsika_output_instance, property, log_z=False):
    """
    The next functions below are used by the the corsikaOutput class to plot all sort of information
    from the Cherenkov photons saved.

    Create the figure of a 2D plot. The parameter `name` indicate which plot. Choices are
    "counts", "density", "direction".

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    property: string
        Name of the quantity. Options are: "counts", "density", "direction", "time_altitude" and
        "num_photons_per_telescope".
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.
    """
    function = getattr(
        corsika_output_instance,
        corsika_output_instance._dict_2D_distributions[property]["function"],
    )
    hist_values, x_edges, y_edges = function()

    all_figs = []
    fig_names = []
    for i_hist, _ in enumerate(x_edges):
        fig, ax = plt.subplots()
        if log_z is True:
            norm = colors.LogNorm(vmin=1, vmax=np.amax([np.amax(hist_values[i_hist]), 2]))
        else:
            norm = None
        mesh = ax.pcolormesh(x_edges[i_hist], y_edges[i_hist], hist_values[i_hist], norm=norm)
        ax.set_xlabel(
            f"{corsika_output_instance._dict_2D_distributions[property]['x edges']} "
            f"({corsika_output_instance._dict_2D_distributions[property]['x edges unit']})"
        )
        ax.set_ylabel(
            f"{corsika_output_instance._dict_2D_distributions[property]['y edges']} "
            f"({corsika_output_instance._dict_2D_distributions[property]['y edges unit']})"
        )
        ax.set_xlim(np.amin(x_edges[i_hist]), np.amax(x_edges[i_hist]))
        ax.set_ylim(np.amin(y_edges[i_hist]), np.amax(y_edges[i_hist]))
        ax.set_facecolor("xkcd:black")
        fig.colorbar(mesh)
        all_figs.append(fig)
        if corsika_output_instance.individual_telescopes is False:
            fig_names.append(
                f"{corsika_output_instance._dict_2D_distributions[property]['file name']}"
                f"_all_tels.png"
            )
        else:
            ax.text(
                0.99,
                0.99,
                "tel. " + str(i_hist),
                ha="right",
                va="top",
                transform=ax.transAxes,
                color="white",
            )
            fig_names.append(
                f"{corsika_output_instance._dict_2D_distributions[property]['file name']}"
                f"_tel_index_{corsika_output_instance.telescope_indices[i_hist]}.png",
            )
        plt.close()

    return all_figs, fig_names


def plot_2D_counts(corsika_output_instance, log_z=True):
    """
    Plot the 2D histogram of the photon positions on the ground.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.

    """
    return _kernel_plot_2D_photons(corsika_output_instance, "counts", log_z=log_z)


def plot_2D_density(corsika_output_instance, log_z=True):
    """
    Plot the 2D histogram of the photon density distribution on the ground.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.

    """
    return _kernel_plot_2D_photons(corsika_output_instance, "density", log_z=log_z)


def plot_2D_direction(corsika_output_instance, log_z=True):
    """
    Plot the 2D histogram of the incoming direction of photons.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.

    """
    return _kernel_plot_2D_photons(corsika_output_instance, "direction", log_z=log_z)


def plot_2D_time_altitude(corsika_output_instance, log_z=True):
    """
    Plot the 2D histogram of the time and altitude where the photon was produced.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.

    """
    return _kernel_plot_2D_photons(corsika_output_instance, "time_altitude", log_z=log_z)


def plot_2D_num_photons_per_telescope(corsika_output_instance, log_z=True):
    """
    Plot the 2D histogram of the number of photons per event and per telescope.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.

    """
    return _kernel_plot_2D_photons(
        corsika_output_instance, "num_photons_per_telescope", log_z=log_z
    )


def _kernel_plot_1D_photons(corsika_output_instance, property, log_y=True):
    """
    Create the figure of a 1D plot. The parameter `name` indicate which plot. Choices are
    "counts", "density", "direction".

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    property: string
        Name of the quantity. Options are: "wavelength", "counts", "density", "time", "altitude",
        "num_photons".
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.

    Raises
    ------
    ValueError
        if `name` is not allowed.
    """
    function = getattr(
        corsika_output_instance,
        corsika_output_instance._dict_1D_distributions[property]["function"],
    )
    hist_values, edges = function()
    all_figs = []
    fig_names = []
    for i_hist, _ in enumerate(edges):
        fig, ax = plt.subplots()
        ax.bar(
            edges[i_hist][:-1],
            hist_values[i_hist],
            align="edge",
            width=np.abs(np.diff(edges[i_hist])),
        )
        ax.set_xlabel(
            f"{corsika_output_instance._dict_1D_distributions[property]['edges']} "
            f"({corsika_output_instance._dict_1D_distributions[property]['edges unit']})"
        )
        ax.set_ylabel("Counts")

        if log_y is True:
            ax.set_yscale("log")
        if corsika_output_instance.individual_telescopes is False:
            fig_names.append(
                f"{corsika_output_instance._dict_1D_distributions[property]['file name']}"
                f"_all_tels.png"
            )
        else:
            fig_names.append(
                f"{corsika_output_instance._dict_1D_distributions[property]['file name']}"
                f"_tel_index_{corsika_output_instance.telescope_indices[i_hist]}.png",
            )
        all_figs.append(fig)
    return all_figs, fig_names


def plot_wavelength_distr(corsika_output_instance, log_y=True):
    """
    Plots the 1D distribution of the photon wavelengths

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.
    """
    return _kernel_plot_1D_photons(corsika_output_instance, "wavelength", log_y=log_y)


def plot_counts_distr(corsika_output_instance, log_y=True):
    """
    Plots the 1D distribution, i.e. the radial distribution, of the photons on the ground.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.
    """
    return _kernel_plot_1D_photons(corsika_output_instance, "counts", log_y=log_y)


def plot_density_distr(corsika_output_instance, log_y=True):
    """
    Plots the photon density distribution on the ground.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.
    """
    return _kernel_plot_1D_photons(corsika_output_instance, "density", log_y=log_y)


def plot_time_distr(corsika_output_instance, log_y=True):
    """
    Plots the distribution times in which the photons were generated in ns.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.
    """
    return _kernel_plot_1D_photons(corsika_output_instance, "time", log_y=log_y)


def plot_altitude_distr(corsika_output_instance, log_y=True):
    """
    Plots the distribution of altitude in which the photons were generated in km.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.
    """
    return _kernel_plot_1D_photons(corsika_output_instance, "altitude", log_y=log_y)


def plot_photon_per_event_distr(corsika_output_instance, log_y=True):
    """
    Plots the distribution of the number of Cherenkov photons per event.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.

    """

    return _kernel_plot_1D_photons(corsika_output_instance, "num_photons_per_event", log_y=log_y)


def plot_photon_per_telescope_distr(corsika_output_instance, log_y=True):
    """
    Plots the distribution of the number of Cherenkov photons per telescope.

    Parameters
    ----------
    corsika_output_instance: corsika.corsika_output.corsikaOutput
        instance of corsika.corsika_output.corsikaOutput.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    list
        List of the figure names.

    """

    return _kernel_plot_1D_photons(
        corsika_output_instance, "num_photons_per_telescope", log_y=log_y
    )
