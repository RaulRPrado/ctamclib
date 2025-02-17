"""Prepare layout for coordinate transformations."""

import json
import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import QTable

import simtools.utils.general as gen
from simtools.data_model import data_reader, schema
from simtools.io_operations import io_handler
from simtools.layout.geo_coordinates import GeoCoordinates
from simtools.layout.telescope_position import TelescopePosition
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.utils import names, value_conversion

__all__ = ["ArrayLayout", "InvalidTelescopeListFileError"]


class InvalidTelescopeListFileError(Exception):
    """Exception for invalid telescope list file."""


class InvalidCoordinateDataTypeError(Exception):
    """Exception for low-precision coordinate data type."""


class ArrayLayout:
    """
    Manage telescope positions at the array layout level.

    Parameters
    ----------
    mongo_db_config: dict
        MongoDB configuration.
    site: str
        Site name or location (e.g., North/South or LaPalma/Paranal)
    model_version: str
        Version of the model (e.g., 6.0.0).
    label: str
        Instance label.
    name: str
        Name of the layout.
    telescope_list_file: str or Path
        Path to the telescope list file.
    telescope_list_metadata_file: str or Path
        Path to telescope list metadata (if not part of telescope_list_file)
    validate: bool
        Validate input file list.
    """

    def __init__(
        self,
        mongo_db_config,
        site,
        model_version,
        label=None,
        name=None,
        telescope_list_file=None,
        telescope_list_metadata_file=None,
        validate=False,
    ):
        """Initialize ArrayLayout."""
        self._logger = logging.getLogger(__name__)

        self.model_version = model_version
        self.label = label
        self.name = name
        self.mongo_db_config = mongo_db_config
        self.site = None if site is None else names.validate_site_name(site)
        self.site_model = None
        self.io_handler = io_handler.IOHandler()
        self.geo_coordinates = GeoCoordinates()

        self._telescope_list = []
        self._corsika_observation_level = None
        self._reference_position_dict = {}
        self._array_center = None

        self._initialize_array_layout(
            telescope_list_file=telescope_list_file,
            telescope_list_metadata_file=telescope_list_metadata_file,
            validate=validate,
        )

    def __len__(self):
        """Return number of telescopes in the layout."""
        return len(self._telescope_list)

    def __getitem__(self, i):
        """Return telescope at list position i."""
        return self._telescope_list[i]

    def _initialize_site_parameters_from_db(self):
        """Initialize site parameters required for transformations using the database."""
        self._logger.debug("Initialize parameters from DB")
        if self.mongo_db_config is None:
            raise ValueError("No database configuration provided")

        self.site_model = SiteModel(
            site=self.site,
            model_version=self.model_version,
            mongo_db_config=self.mongo_db_config,
        )
        self._corsika_observation_level = self.site_model.get_corsika_site_parameters().get(
            "corsika_observation_level", None
        )
        self._reference_position_dict = self.site_model.get_reference_point()
        self._logger.debug(f"Reference point: {self._reference_position_dict}")

    def _initialize_coordinate_systems(self):
        """
        Initialize array center and coordinate systems.

        By definition, the array center is at (0,0) in
        the ground coordinate system.

        Raises
        ------
        TypeError
            invalid array center definition.

        """
        self._array_center = TelescopePosition()
        self._array_center.name = "array_center"
        self._array_center.set_coordinates("ground", 0.0 * u.m, 0.0 * u.m)
        self._set_array_center_utm()
        self._array_center.set_altitude(
            u.Quantity(self._reference_position_dict.get("center_altitude", np.nan * u.m))
        )
        _name = self._reference_position_dict.get("array_name")
        self.name = _name if _name is not None else self.name

        self._logger.debug(f"Initialized array center at UTM {self._reference_position_dict}")
        self._array_center.convert_all(
            crs_local=self.geo_coordinates.crs_local(self._array_center),
            crs_wgs84=self.geo_coordinates.crs_wgs84(),
            crs_utm=self.geo_coordinates.crs_utm(
                self._reference_position_dict.get("epsg_code", None)
            ),
        )

    def _set_array_center_utm(self):
        """
        Set array center coordinates in UTM system.

        Convert array center position to WGS84 system
        (as latitudes are required for the definition
        for the definition of the ground coordinate system)

        """
        self._array_center.set_coordinates(
            "utm",
            u.Quantity(self._reference_position_dict.get("center_easting", np.nan * u.m)),
            u.Quantity(self._reference_position_dict.get("center_northing", np.nan * u.m)),
        )
        self._array_center.convert_all(
            crs_local=None,
            crs_wgs84=self.geo_coordinates.crs_wgs84(),
            crs_utm=self.geo_coordinates.crs_utm(
                self._reference_position_dict.get("epsg_code", None)
            ),
        )

    def _altitude_from_corsika_z(self, pos_z=None, altitude=None, telescope_axis_height=None):
        """
        Calculate altitude.

        The value is calculated from CORSIKA z-coordinate (if pos_z is given) or CORSIKA
        z-coordinate from altitude (if altitude is given).

        Parameters
        ----------
        pos_z: astropy.Quantity
            CORSIKA z-coordinate of telescope in equivalent units of meter.
        altitude: astropy.Quantity
            Telescope altitude in equivalent units of meter.
        tel_axis_height: astropy.Quantity
            Telescope axis height in equivalent units of meter.

        Returns
        -------
        astropy.Quantity
            Altitude or CORSIKA z-coordinate (np.nan in case of ill-defined value).

        """
        self._logger.debug(
            f"pos_z: {pos_z}, altitude: {altitude}, "
            f"axis_height: {telescope_axis_height}, "
            f"obs_level: {self._corsika_observation_level}"
        )

        if pos_z is not None and altitude is None:
            return TelescopePosition.convert_telescope_altitude_from_corsika_system(
                pos_z,
                self._corsika_observation_level,
                telescope_axis_height,
            )

        if altitude is not None and pos_z is None:
            return TelescopePosition.convert_telescope_altitude_to_corsika_system(
                altitude,
                self._corsika_observation_level,
                telescope_axis_height,
            )
        return np.nan

    def _load_telescope_names(self, row):
        """
        Read and set telescope names.

        Parameters
        ----------
        row: astropy table row
            table row.

        Returns
        -------
        tel: TelescopePosition
            Instance of TelescopePosition.

        Raises
        ------
        InvalidTelescopeListFileError
            in case neither telescope name or asset_code / sequence number are given.

        """
        tel = TelescopePosition()
        try:
            tel.name = row["telescope_name"]
            if "asset_code" not in row:
                try:
                    tel.asset_code = names.get_array_element_type_from_name(tel.name)
                # asset code is not a valid telescope name; possibly a calibration device
                except ValueError:
                    tel.asset_code = tel.name.split("-")[0]
        except KeyError:
            pass
        try:
            if tel.name is None:
                tel.name = row["asset_code"] + "-" + row["sequence_number"]
            tel.asset_code = row["asset_code"]
            tel.sequence_number = row["sequence_number"]
        except KeyError:
            pass
        try:
            tel.geo_code = row["geo_code"]
        except KeyError:
            pass
        if tel.name is None:
            msg = "Missing required row with telescope_name or asset_code/sequence_number"
            self._logger.error(msg)
            raise InvalidTelescopeListFileError(msg)

        return tel

    def _try_set_coordinate(self, row, tel, table, crs_name, key1, key2):
        """
        Try and set coordinates for all coordinate systems.

        Parameters
        ----------
        row: dict
            A row of the astropy.table.Table with array element coordinates.
        tel: TelescopePosition
            Instance of TelescopePosition.
        table: astropy.table.Table or astropy.table.QTable
            data table with array element coordinates.
        crs_name: str
            Name of coordinate system.
        key1: str
            Name of x-coordinate.
        key2: str
            Name of y-coordinate.
        """
        try:
            tel.set_coordinates(
                crs_name,
                value_conversion.get_value_as_quantity(row[key1], table[key1].unit),
                value_conversion.get_value_as_quantity(row[key2], table[key2].unit),
            )
        except KeyError:
            pass

    def _try_set_altitude(self, row, tel, table):
        """
        Try and set altitude.

        It sets the altitude of the TelescopePosition instance.

        Parameters
        ----------
        row: dict
            A row of the astropy.table.Table with array element coordinates.
        tel: TelescopePosition
            Instance of TelescopePosition.
        table: astropy.table.Table or astropy.table.QTable
            data table with array element coordinates.
        """
        try:
            tel.set_altitude(
                self._altitude_from_corsika_z(
                    pos_z=value_conversion.get_value_as_quantity(
                        row["position_z"], table["position_z"].unit
                    ),
                    telescope_axis_height=tel.get_axis_height(),
                )
            )
        except KeyError:
            pass
        try:
            tel.set_altitude(
                value_conversion.get_value_as_quantity(row["altitude"], table["altitude"].unit)
            )
        except KeyError:
            pass

    def _initialize_array_layout(
        self, telescope_list_file, telescope_list_metadata_file=None, validate=False
    ):
        """
        Initialize the Layout array including site and telescope parameters.

        Read array list if telescope_list_file is given.

        Parameters
        ----------
        telescope_list_file: str or Path
            Path to the telescope list file.
        telescope_list_metadata_file: str or Path
            Path to the telescope list metadata file.
        validate: bool
            Validate telescope list file against schema.

        Returns
        -------
        astropy.table.QTable
            Table with the telescope layout information.
        """
        self._logger.debug("Initializing array (site and telescope parameters)")
        self._initialize_site_parameters_from_db()
        self._initialize_coordinate_systems()

        if telescope_list_file is None:
            return None

        self._logger.debug(f"Reading telescope list from {telescope_list_file}")
        if Path(telescope_list_file).suffix == ".json":
            table = self._read_table_from_json_file(file_name=telescope_list_file)
        else:
            table = data_reader.read_table_from_file(
                file_name=telescope_list_file,
                validate=validate,
                metadata_file=telescope_list_metadata_file,
            )

        for row in table:
            tel = self._load_telescope_names(row)
            if names.get_collection_name_from_array_element_name(tel.name) == "telescopes":
                self._set_telescope_auxiliary_parameters(tel)
            self._try_set_coordinate(row, tel, table, "ground", "position_x", "position_y")
            self._try_set_coordinate(row, tel, table, "utm", "utm_east", "utm_north")
            self._try_set_coordinate(row, tel, table, "mercator", "latitude", "longitude")
            self._try_set_altitude(row, tel, table)
            self._telescope_list.append(tel)

        return table

    def _read_table_from_json_file(self, file_name):
        """
        Read a telescope position from a json file and return as astropy table.

        Parameters
        ----------
        file_name: str or Path
            Path to the json file.

        Returns
        -------
        astropy.table.QTable
            Table with the telescope layout information.
        """
        with Path(file_name).open("r", encoding="utf-8") as file:
            data = json.load(file)

        position = data["value"]
        if isinstance(position, str):
            position = gen.convert_string_to_list(position)
        self.site = data.get("site", None)

        table = QTable()
        table["telescope_name"] = [data["instrument"]]
        if "utm" in data["parameter"]:
            table["utm_east"] = [position[0]] * u.Unit(data["unit"])
            table["utm_north"] = [position[1]] * u.Unit(data["unit"])
            table["altitude"] = [position[2]] * u.Unit(data["unit"])
        else:
            table["position_x"] = [position[0]] * u.Unit(data["unit"])
            table["position_y"] = [position[1]] * u.Unit(data["unit"])
            table["position_z"] = [position[2]] * u.Unit(data["unit"])
        return table

    def _get_telescope_model(self, telescope_name):
        """
        Get telescope model from the database.

        Parameters
        ----------
        telescope_name: str
            Name of the telescope.

        Returns
        -------
        TelescopeModel
            Telescope model instance.
        """
        return TelescopeModel(
            site=self.site,
            telescope_name=telescope_name,
            model_version=self.model_version,
            mongo_db_config=self.mongo_db_config,
            label=self.label,
        )

    def _set_telescope_auxiliary_parameters(self, telescope, telescope_name=None):
        """
        Set auxiliary CORSIKA parameters for a telescope.

        Uses as default the design model if telescope is not found in the database.

        Parameters
        ----------
        telescope: TelescopePosition
            Instance of TelescopePosition.

        """
        telescope_name = telescope_name if telescope_name is not None else telescope.name
        if names.get_collection_name_from_array_element_name(telescope_name) == "telescopes":
            self._logger.debug(
                f"Reading auxiliary telescope parameters for {telescope_name}"
                f" (model version {self.model_version})"
            )
            try:
                tel_model = self._get_telescope_model(telescope_name)
            except ValueError:
                tel_model = self._get_telescope_model(
                    names.get_design_model_from_name(telescope_name),
                )

            for para in ("telescope_axis_height", "telescope_sphere_radius"):
                telescope.set_auxiliary_parameter(
                    para, tel_model.get_parameter_value_with_unit(para)
                )

    def add_telescope(
        self, telescope_name, crs_name, xx, yy, altitude=None, tel_corsika_z=None, design_model=None
    ):
        """
        Add an individual telescope to the telescope list.

        Parameters
        ----------
        telescope_name: str
            Name of the telescope starting with L, M or S (e.g. LST-01, MST-06 ...)
        crs_name: str
            Name of coordinate system
        xx: astropy.units.quantity.Quantity
            x-coordinate for the given coordinate system
        yy: astropy.units.quantity.Quantity
            y-coordinate for the given coordinate system
        altitude: astropy.units.quantity.Quantity
            Altitude coordinate in equivalent units of u.m.
        tel_corsika_z: astropy.units.quantity.Quantity
            CORSIKA z-position (requires setting of CORSIKA observation level and telescope sphere\
            center).
        design_model: str
            Name of the design model (optional).
            If none, telescope type + "-design" is used.
        """
        tel = TelescopePosition(name=telescope_name)
        self._set_telescope_auxiliary_parameters(tel, design_model)
        tel.set_coordinates(crs_name, xx, yy)
        if altitude is not None:
            tel.set_altitude(altitude)
        elif tel_corsika_z is not None:
            tel.set_altitude(
                self._altitude_from_corsika_z(
                    pos_z=tel_corsika_z, telescope_axis_height=tel.get_axis_height()
                )
            )
        self._telescope_list.append(tel)

    def _get_export_metadata(self):
        """
        File metadata for export of array element list to file.

        Included array center definition, CORSIKA telescope parameters, and EPSG code.

        Returns
        -------
        dict
            Metadata header for array element list export.

        """
        _meta = {}
        if self._array_center:
            (
                _meta["center_easting"],
                _meta["center_northing"],
                _meta["center_altitude"],
            ) = self._array_center.get_coordinates("utm")
        _meta["epsg_code"] = self._reference_position_dict.get("epsg_code", None)
        _meta["array_name"] = self.name

        return _meta

    def export_telescope_list_table(self, crs_name):
        """
        Export array elements positions to astropy table.

        Parameters
        ----------
        crs_name: str
            Name of coordinate system to be used for export.

        Returns
        -------
        astropy.table.QTable
            Astropy table with the telescope layout information.
        """
        table = QTable(meta=self._get_export_metadata())

        tel_names, asset_code, sequence_number, geo_code = [], [], [], []
        pos_x, pos_y, pos_z, pos_t, tel_r = [], [], [], [], []
        for tel in self._telescope_list:
            tel_names.append(tel.name)
            asset_code.append(tel.asset_code)
            sequence_number.append(tel.sequence_number)
            geo_code.append(tel.geo_code)
            x, y, z = tel.get_coordinates(crs_name)
            if crs_name == "ground":
                z = self._altitude_from_corsika_z(
                    altitude=z, telescope_axis_height=tel.get_axis_height()
                )
                pos_t.append(tel.get_axis_height())
            pos_x.append(x)
            pos_y.append(y)
            pos_z.append(z)
            tel_r.append(tel.get_sphere_radius())

        # prefer asset_code / sequence_number of telescope_name
        if all(v is not None for v in asset_code) and all(v is not None for v in sequence_number):
            table["asset_code"] = asset_code
            table["sequence_number"] = sequence_number
        else:
            table["telescope_name"] = tel_names
        if any(v is not None for v in geo_code):
            table["geo_code"] = geo_code

        try:
            _name_x, _name_y, _name_z = self._telescope_list[0].get_coordinates(
                crs_name=crs_name, coordinate_field="name"
            )
            table[_name_x] = pos_x
            table[_name_y] = pos_y
            table[_name_z] = pos_z
            if len(pos_t) > 0:
                table["telescope_axis_height"] = pos_t
            if len(tel_r) > 0:
                table["sphere_radius"] = tel_r
        except IndexError:
            pass

        if "telescope_name" in table.colnames:
            table.sort("telescope_name")
        if "asset_code" in table.colnames:
            table.sort(["asset_code", "sequence_number"])

        return table

    def export_one_telescope_as_json(
        self,
        crs_name,
        parameter_version=None,
        schema_version=None,
    ):
        """
        Return a list containing a single telescope in simtools-DB-style json.

        Parameters
        ----------
        crs_name: str
            Name of coordinate system to be used for export.
        schema_version: str
            Version of the schema.

        Returns
        -------
        dict
            Dictionary with array element information.
        """
        table = self.export_telescope_list_table(crs_name)
        if len(table) != 1:
            raise ValueError("Only one telescope can be exported to json")
        parameter_name = value = None
        if crs_name == "ground":
            parameter_name = "array_element_position_ground"
            value = [
                table["position_x"][0].value,
                table["position_y"][0].value,
                table["position_z"][0].value,
            ]
        elif crs_name == "utm":
            parameter_name = "array_element_position_utm"
            value = [
                table["utm_east"][0].value,
                table["utm_north"][0].value,
                table["altitude"][0].value,
            ]
        elif crs_name == "mercator":
            parameter_name = "array_element_position_mercator"
            value = [
                table["latitude"][0].value,
                table["longitude"][0].value,
                table["altitude"][0].value,
            ]

        return {
            "schema_version": schema.get_model_parameter_schema_version(schema_version),
            "parameter": parameter_name,
            "instrument": table["telescope_name"][0],
            "site": self.site,
            "parameter_version": parameter_version,
            "unique_id": None,
            "value": value,
            "unit": "m",
            "type": "float64",
            "file": False,
        }

    def get_number_of_telescopes(self):
        """
        Return the number of telescopes in the list.

        Returns
        -------
        int
            Number of telescopes.
        """
        return len(self._telescope_list)

    def print_telescope_list(self, crs_name):
        """
        Print list of telescopes.

        Parameters
        ----------
        crs_name: str
            Name of coordinate system to be used for export.

        """
        for tel in self._telescope_list:
            tel.print_compact_format(
                crs_name=crs_name,
                print_header=(tel == self._telescope_list[0]),
                corsika_observation_level=(
                    self._corsika_observation_level if crs_name == "ground" else None
                ),
            )

    def convert_coordinates(self):
        """Perform all the possible conversions the coordinates of the tel positions."""
        self._logger.info("Converting telescope coordinates")

        crs_wgs84 = self.geo_coordinates.crs_wgs84()
        crs_local = self.geo_coordinates.crs_local(self._array_center)
        crs_utm = self.geo_coordinates.crs_utm(self._reference_position_dict.get("epsg_code", None))

        for tel in self._telescope_list:
            tel.convert_all(
                crs_local=crs_local,
                crs_wgs84=crs_wgs84,
                crs_utm=crs_utm,
            )

    def select_assets(self, asset_list=None):
        """
        Select a subsets of telescopes / assets from the layout.

        Parameters
        ----------
        asset_list: list
            List of assets to be selected (telescope names or types)

        Raises
        ------
        ValueError
            If the asset list is empty.

        """
        _n_telescopes = len(self._telescope_list)
        try:
            if len(asset_list) > 0:
                _telescope_list_from_name = [
                    tel for tel in self._telescope_list if tel.asset_code in asset_list
                ]
                _telescope_list_from_type = [
                    tel
                    for tel in self._telescope_list
                    if names.get_array_element_type_from_name(tel.asset_code) in asset_list
                ]
                self._telescope_list = list(
                    set(_telescope_list_from_name + _telescope_list_from_type)
                )
            self._logger.info(
                f"Selected {len(self._telescope_list)} telescopes"
                f" (from originally {_n_telescopes})"
            )
        except TypeError:
            self._logger.info("No asset list provided, keeping all telescopes")
