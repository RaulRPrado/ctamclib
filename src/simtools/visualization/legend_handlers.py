"""Helper functions for legend handlers used for plotting."""

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

__all__ = [
    "EdgePixelObject",
    "HexEdgePixelHandler",
    "HexOffPixelHandler",
    "HexPixelHandler",
    "LSTHandler",
    "LSTObject",
    "MSTHandler",
    "MSTObject",
    "MeanRadiusOuterEdgeHandler",
    "MeanRadiusOuterEdgeObject",
    "OffPixelObject",
    "PixelObject",
    "SCTHandler",
    "SCTObject",
    "SSTHandler",
    "SSTObject",
    "SquareEdgePixelHandler",
    "SquareOffPixelHandler",
    "SquarePixelHandler",
    "TelescopeHandler",
]

# assume same size of telescope at both sides
# (good assumption for plotting)
lst = "LSTN"
mst = "MSTN"
sst = "SSTS"
sct = "SCTS"
hess = "HESS"
magic = "MAGIC"
veritas = "VERITAS"


def calculate_center(handlebox, width_factor=3, height_factor=3):
    """
    Calculate the center of the handlebox based on given factors.

    Parameters
    ----------
    handlebox: matplotlib.legend.Legend
        The handlebox object from the legend.
    width_factor: int, optional
        The factor to adjust the width.
    height_factor: int, optional
        The factor to adjust the height.

    Returns
    -------
    tuple
        The calculated (x0, y0) position.
    """
    x0 = handlebox.xdescent + handlebox.width / width_factor
    y0 = handlebox.ydescent + handlebox.height / height_factor
    return x0, y0


class TelescopeHandler:
    """
    Telescope handler that centralizes the telescope information.

    Individual telescopes handlers inherit from this class.
    """

    def __init__(self, radius=None):
        self.colors_dict = {
            "LSTN": "darkorange",
            "MSTx": "dodgerblue",
            "MSTN": "dodgerblue",
            "LSTS": "darkorange",
            "MSTS": "dodgerblue",
            "SCTS": "black",
            "SSTS": "darkgreen",
            "HESS": "grey",
            "MAGIC": "grey",
            "VERITAS": "grey",
        }

        # hardwired values; this is for plotting purposes only
        self.radius_dict = {
            "LSTN": 12.5,
            "MSTx": 9.15,
            "MSTN": 9.15,
            "LSTS": 12.5,
            "MSTS": 9.15,
            "SCTS": 7.15,
            "SSTS": 3.0,
            "HESS": 6.0,
            "MAGIC": 8.5,
            "VERITAS": 6.0,
        }


class PixelObject:
    """Pixel Object."""


class EdgePixelObject:
    """Edge-Pixel Object."""


class OffPixelObject:
    """Off-Pixel Object."""


class LSTObject:
    """LST Object."""


class MSTObject:
    """MST Object."""


class SCTObject:
    """SCT Object."""


class SSTObject:
    """SST Object."""


class HESSObject:
    """HESS Object."""


class MAGICObject:
    """MAGIC Object."""


class VERITASObject:
    """VERITAS Object."""


class MeanRadiusOuterEdgeObject:
    """Object for Mean radius outer edge."""


class HexPixelHandler:
    """Legend handler class to plot a hexagonal "on" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):  # noqa: D102
        x0, y0 = calculate_center(handlebox)
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * handlebox.height,
            orientation=np.deg2rad(30),
            facecolor=(1, 1, 1, 0),
            edgecolor=(0, 0, 0, 1),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class HexEdgePixelHandler:
    """Legend handler class to plot a hexagonal "edge" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):  # noqa: D102
        x0, y0 = calculate_center(handlebox)
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * handlebox.height,
            orientation=np.deg2rad(30),
            facecolor=(*mcolors.to_rgb("brown"), 0.5),
            edgecolor=(*mcolors.to_rgb("black"), 1),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class HexOffPixelHandler:
    """Legend handler class to plot a hexagonal "off" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):  # noqa: D102
        x0, y0 = calculate_center(handlebox)
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * handlebox.height,
            orientation=np.deg2rad(30),
            facecolor="black",
            edgecolor="black",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SquarePixelHandler:
    """Legend handler class to plot a square "on" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):  # noqa: D102
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width = height = handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=(1, 1, 1, 0),
            edgecolor=(0, 0, 0, 1),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SquareEdgePixelHandler:
    """Legend handler class to plot a square "edge" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):  # noqa: D102
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width = height = handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=(*mcolors.to_rgb("brown"), 0.5),
            edgecolor=(*mcolors.to_rgb("black"), 1),
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SquareOffPixelHandler:
    """Legend handler class to plot a square "off" pixel."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):  # noqa: D102
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width = height = handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor="black",
            edgecolor="black",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class LSTHandler(TelescopeHandler):
    """Legend handler class to plot a representation of an LST in an array layout."""

    def legend_artist(self, _, __, ___, handlebox):  # noqa: D102
        x0, y0 = calculate_center(handlebox, 10 / 3, 2)
        radius = handlebox.height
        patch = mpatches.Circle(
            xy=(x0, y0),
            radius=radius * self.radius_dict[lst] / self.radius_dict[lst],
            facecolor="none",
            edgecolor=self.colors_dict[lst],
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class MSTHandler(TelescopeHandler):
    """Legend handler class to plot a representation of an MST in an array layout."""

    def legend_artist(self, _, __, ___, handlebox):  # noqa: D102
        x0, y0 = calculate_center(handlebox, 4, 2)
        radius = handlebox.height
        patch = mpatches.Circle(
            xy=(x0, y0),
            radius=radius * self.radius_dict[mst] / self.radius_dict[lst],
            facecolor=self.colors_dict[mst],
            edgecolor=self.colors_dict[mst],
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SSTHandler(TelescopeHandler):
    """Legend handler class to plot a representation of an SST in an array layout."""

    def legend_artist(self, _, __, ___, handlebox):  # noqa: D102
        x0, y0 = calculate_center(handlebox, 4, 2)
        radius = handlebox.height
        patch = mpatches.Circle(
            xy=(x0, y0),
            radius=radius * self.radius_dict[sst] / self.radius_dict[lst],
            facecolor=self.colors_dict[sst],
            edgecolor=self.colors_dict[sst],
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class SCTHandler(TelescopeHandler):
    """Legend handler class to plot a representation of an SCT in an array layout."""

    def legend_artist(self, _, __, ___, handlebox):  # noqa: D102
        x0, y0 = calculate_center(handlebox, 10, 1)
        width = height = handlebox.height
        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=self.colors_dict[sct],
            edgecolor=self.colors_dict[sct],
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class HESSHandler(TelescopeHandler):
    """Legend handler class to plot a representation of an HESS in an array layout."""

    def legend_artist(self, _, __, ___, handlebox):
        x0, y0 = calculate_center(handlebox)
        radius = handlebox.height
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * radius * self.radius_dict[hess] / self.radius_dict[lst],
            orientation=np.deg2rad(30),
            facecolor=self.colors_dict[hess],
            edgecolor=self.colors_dict[hess],
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class MAGICHandler(TelescopeHandler):
    """Legend handler class to plot a representation of an MAGIC in an array layout."""

    def legend_artist(self, _, __, ___, handlebox):
        x0, y0 = calculate_center(handlebox)
        radius = handlebox.height
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * radius * self.radius_dict[magic] / self.radius_dict[lst],
            orientation=np.deg2rad(30),
            facecolor=self.colors_dict[magic],
            edgecolor=self.colors_dict[magic],
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class VERITASHandler(TelescopeHandler):
    """Legend handler class to plot a representation of an VERITAS in an array layout."""

    def legend_artist(self, _, __, ___, handlebox):
        x0, y0 = calculate_center(handlebox)
        radius = handlebox.height
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.7 * radius * self.radius_dict[veritas] / self.radius_dict[lst],
            orientation=np.deg2rad(30),
            facecolor=self.colors_dict[veritas],
            edgecolor=self.colors_dict[veritas],
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


class MeanRadiusOuterEdgeHandler:
    """Legend handler class to plot a the mean radius outer edge of the dish."""

    @staticmethod
    def legend_artist(_, __, ___, handlebox):  # noqa: D102
        x0, y0 = calculate_center(handlebox, 4, 4)
        radius = handlebox.height
        patch = mpatches.Circle(
            xy=(x0, y0),
            radius=radius,
            facecolor="none",
            edgecolor="darkorange",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        return patch


all_telescope_objects = {
    "LSTN": LSTObject,
    "LSTS": LSTObject,
    "MSTN": MSTObject,
    "MSTS": MSTObject,
    sct: SCTObject,
    sst: SSTObject,
    hess: HESSObject,
    magic: MAGICObject,
    veritas: VERITASObject,
}
all_telescope_handlers = {
    "LSTN": LSTHandler,
    "LSTS": LSTHandler,
    "MSTN": MSTHandler,
    "MSTS": MSTHandler,
    sct: SCTHandler,
    sst: SSTHandler,
    hess: HESSHandler,
    magic: MAGICHandler,
    veritas: VERITASHandler,
}
legend_handler_map = {}
try:
    for tel_type in all_telescope_objects.keys():
        legend_handler_map[all_telescope_objects[tel_type]] = all_telescope_handlers[tel_type]
except KeyError:
    pass
