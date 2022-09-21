import logging

import astropy.units as u
import numpy as np
import pyproj
from astropy.table import Table

import simtools.config as cfg
import simtools.io_handler as io
from simtools.layout.telescope_position import TelescopePosition
from simtools.util import names


class InvalidTelescopeListFile(Exception):
    pass


class LayoutArray:
    """
    Manage telescope positions at the array layout level.

    Methods
    -------
    fromLayoutArrayName(layoutArrayName, label=None)
        Create a LayoutArray from a layout name (e.g. South-4LST, North-Prod5, ...)
    readTelescopeListFile(telescopeListFile)
        Read list of telescopes from a ecsv file.
    addTelescope(
        telescopeName,
        crsName,
        xx,
        yy,
        altitude=None
        telCorsikaZ=None
    )
        Add an individual telescope to the telescope list.
    exportTelescopeList(filesLocation)
        Export a ECSV file with the telescope positions.
    getNumberOfTelescopes()
        Return the number of telescopes in the list.
    getCorsikaInputList()
        Get a string with the piece of text to be added to
        the CORSIKA input file.
    printTelescopeList()
        Print list of telescopes in current layout for inspection.
    convertCoordinates()
        Perform all the possible conversions the coordinates of the tel positions.
    """

    def __init__(
        self,
        label=None,
        name=None,
        layoutCenterData=None,
        corsikaTelescopeData=None,
    ):
        """
        LayoutArray init.

        Parameters
        ----------
        name: str
            Name of the telescope (e.g L-01, S-05, ...)
        label: str
            Instance label.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        layoutCenterData: dict
            Dict describing array center coordinates.
        corsikaTelescopeData: dict
            Dict describing CORSIKA telescope parameters.

        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init LayoutArray")

        self.label = label

        self.name = name
        self._telescopeList = []
        self._epsg = None
        if layoutCenterData:
            self._initalizeCoordinateSystems(layoutCenterData)
        else:
            self._initalizeCoordinateSystems(self._layoutCenterDefaults(), defaults_init=True)
        self._corsikaTelescope = {}
        if corsikaTelescopeData:
            self._initializeCorsikaTelescope(corsikaTelescopeData)
        else:
            self._initializeCorsikaTelescope(self._corsikaTelescopeDefault())

    @classmethod
    def fromLayoutArrayName(cls, layoutArrayName, label=None):
        """
        Create a LayoutArray from a layout name (e.g. South-4LST, North-Prod5, ...)

        Parameters
        ----------
        layoutArrayName: str
            e.g. South-4LST, North-Prod5 ...
        label: str, optional
            Instance label. Important for output file naming.

        Returns
        -------
        Instance of the LayoutArray class.
        """

        spl = layoutArrayName.split("-")
        siteName = names.validateSiteName(spl[0])
        arrayName = names.validateLayoutArrayName(spl[1])
        validLayoutArrayName = siteName + "-" + arrayName

        layout = cls(name=validLayoutArrayName, label=label)

        telescopeListFile = io.getDataFile(
            "layout", "telescope_positions-{}.ecsv".format(validLayoutArrayName)
        )
        layout.readTelescopeListFile(telescopeListFile)

        return layout

    def __len__(self):
        return len(self._telescopeList)

    def __getitem__(self, i):
        return self._telescopeList[i]

    def _initializeCorsikaTelescope(self, corsika_dict):
        """
        Initialize CORSIKA telescope parameters.

        Parameters
        ----------
        corsika_dict dict
            dictionary with CORSIKA telescope parameters

        """
        try:
            if corsika_dict["corsika_obs_level"]:
                self._corsikaTelescope["corsika_obs_level"] = u.Quantity(
                    corsika_dict["corsika_obs_level"]
                )
            else:
                self._corsikaTelescope["corsika_obs_level"] = None
            self._corsikaTelescope["corsika_sphere_center"] = {}
            for key, value in corsika_dict["corsika_sphere_center"].items():
                self._corsikaTelescope["corsika_sphere_center"][key] = u.Quantity(value)
            self._corsikaTelescope["corsika_sphere_radius"] = {}
            for key, value in corsika_dict["corsika_sphere_radius"].items():
                self._corsikaTelescope["corsika_sphere_radius"][key] = u.Quantity(value)
        except KeyError:
            pass

    def _initalizeCoordinateSystems(self, center_dict, defaults_init=False):
        """
        Initialize array center and coordinate systems.

        Parameters
        ----------
        center_dict: dict
            dictionary with coordinates of array center.
        defaults_init: bool
            default initialisation without transformation in all available projections.

        Raises
        ------
        TypeError
            invalid array center definition

        """

        self._arrayCenter = TelescopePosition()
        self._arrayCenter.name = "array_center"

        self._arrayCenter.setCoordinates("corsika", 0 * u.m, 0 * u.m, 0.0 * u.m)

        try:
            self._arrayCenter.setCoordinates(
                "mercator",
                u.Quantity(center_dict["center_lat"]),
                u.Quantity(center_dict["center_lon"]),
            )
        except (TypeError, KeyError):
            pass
        try:
            self._epsg = center_dict["EPSG"]
            self._arrayCenter.setCoordinates(
                "utm",
                u.Quantity(center_dict["center_easting"]),
                u.Quantity(center_dict["center_northing"]),
            )
        except (TypeError, KeyError):
            pass
        try:
            self._arrayCenter.setAltitude(u.Quantity(center_dict["center_alt"]))
        except (TypeError, KeyError):
            pass

        if not defaults_init:
            self._arrayCenter.convertAll(
                crsLocal=self._getCrsLocal(), crsWgs84=self._getCrsWgs84(), crsUtm=self._getCrsUtm()
            )

    def _altitudeFromCorsikaZ(self, pos_z, tel_name):
        """
        Calculate altitude from CORSIKA z-coordinate.

        Parameters
        ----------
        pos_z
            CORSIKA z-coordinate of telescope
        tel_name: str
            Telescope Name

        Returns
        -------
        astropy.Quantity
            Altitude (np.nan in case of ill-defined value)

        """
        if self.getTelescopeType(tel_name) is None:
            return np.nan

        try:
            return TelescopePosition.convertTelescopeAltitudeFromCorsikaSystem(
                pos_z,
                self._corsikaTelescope["corsika_obs_level"],
                self._corsikaTelescope["corsika_sphere_center"][self.getTelescopeType(tel_name)],
            )
        except KeyError:
            self._logger.error("Missing definition of CORSIKA sphere center")
            raise

    def _loadTelescopeList(self, table):
        """
        Load list of telescope from an astropy table

        Parameters
        ----------
        table: astropy.table
            data table with array element coordinates

        """

        for row in table:
            tel = TelescopePosition()

            try:
                tel.name = row["telescope_name"]
                if "asset_code" not in row:
                    tel.asset_code = self.getTelescopeType(tel.name)
            except KeyError:
                pass
            try:
                tel.name = row["asset_code"] + "-" + row["sequence_number"]
                tel.asset_code = row["asset_code"]
                tel.sequence_number = row["sequence_number"]
            except KeyError:
                pass
            if tel.name is None:
                msg = "Missing required row with telescope_name or asset_code/sequence_number"
                self._logger.error(msg)
                raise InvalidTelescopeListFile(msg)

            try:
                tel.geo_code = row["geo_code"]
            except KeyError:
                pass

            try:
                tel.name = row["telescope_name"]
                tel.setCoordinates(
                    "corsika",
                    row["pos_x"] * table["pos_x"].unit,
                    row["pos_y"] * table["pos_y"].unit,
                )
            except KeyError:
                pass
            try:
                tel.setCoordinates(
                    "utm",
                    row["utm_east"] * table["utm_east"].unit,
                    row["utm_north"] * table["utm_north"].unit,
                )
            except KeyError:
                pass
            try:
                tel.setCoordinates(
                    "mercator",
                    row["lat"] * table["lat"].unit,
                    row["lon"] * table["lon"].unit,
                )
            except KeyError:
                pass
            try:
                tel.setAltitude(
                    self._altitudeFromCorsikaZ(row["pos_z"] * table["pos_z"].unit, tel.name)
                )
            except KeyError:
                pass
            try:
                tel.setAltitude(row["alt"] * table["alt"].unit)
            except KeyError:
                pass

            self._telescopeList.append(tel)

    def readTelescopeListFile(self, telescopeListFile):
        """
        Read list of telescopes from a ecsv file.

        Parameters
        ----------
        telescopeListFile: str (or Path)
            Path to the telescope list file.

        Raises
        ------
        InvalidTelescopeListFile
            If cannot read telescope list file or the table does not contain
            telescope_name key.

        """
        try:
            table = Table.read(telescopeListFile, format="ascii.ecsv")
        except FileNotFoundError:
            logging.error("Error reading list of array elements from {}".format(telescopeListFile))
            raise

        self._logger.info("Reading array elements from {}".format(telescopeListFile))

        self._initializeCorsikaTelescope(table.meta)
        self._initalizeCoordinateSystems(table.meta)
        self._loadTelescopeList(table)

    def addTelescope(
        self,
        telescopeName,
        crsName,
        xx,
        yy,
        altitude=None,
        telCorsikaZ=None,
    ):
        """
        Add an individual telescope to the telescope list.

        Parameters
        ----------
        telescopeName: str
            Name of the telescope starting with L, M or S (e.g. L-01, M-06 ...)
        crsName: str
            Name of coordinate system
        xx: astropy.units.quantity.Quantity
            x-coordination for the given coordinate system
        yy: astropy.units.quantity.Quantity
            y-coordination for the given coordinate system
        altitude: astropy.units.quantity.Quantity
            Altitude coordinate in equivalent units of u.m.
        telCorsikaZ: astropy.units.quantity.Quantity
            CORSIKA z-position (requires setting of CORSIKA observation level and telescope
            sphere center).
        """

        tel = TelescopePosition(name=telescopeName)
        tel.setCoordinates(crsName, xx, yy)
        if altitude is not None:
            tel.setAltitude(altitude)
        elif telCorsikaZ is not None:
            tel.setAltitude(self._altitudeFromCorsikaZ(telCorsikaZ, tel.name))
        self._telescopeList.append(tel)

    def _get_export_metadata(self, export_corsika_meta=False):
        """
        File metadata for export of array element list to file.
        Included array center definiton, CORSIKA telescope parameters,
        and EPSG centre

        Parameters
        ----------
        export_corsika_meta: bool
            write CORSIKA metadata

        Returns
        -------
        dict
            metadata header for array element list export

        """

        _meta = {}

        _meta["center_lon"] = None
        _meta["center_lat"] = None
        _meta["center_northing"] = None
        _meta["center_easting"] = None
        _meta["center_alt"] = None
        if self._arrayCenter:
            _meta["center_lat"], _meta["center_lon"], _ = self._arrayCenter.getCoordinates(
                "mercator"
            )
            (
                _meta["center_easting"],
                _meta["center_northing"],
                _meta["center_alt"],
            ) = self._arrayCenter.getCoordinates("utm")
        if export_corsika_meta:
            _meta.update(self._corsikaTelescope)
        _meta["EPSG"] = self._epsg

        return _meta

    def exportTelescopeList(self, crsName, filesLocation=None):
        """
        Export array elements positions to ECSV file

        Parameters
        ----------
        crsName: str
            Name of coordinate system to be used for export.
        filesLocation: str (or Path), optional
            Directory for output. If not given, taken from config file.

        """

        fileName = names.layoutTelescopeListFileName(self.name, None)

        self._filesLocation = cfg.getConfigArg("outputLocation", filesLocation)
        self._outputDirectory = io.getLayoutOutputDirectory(self._filesLocation, self.label)
        self._outputDirectory.mkdir(parents=True, exist_ok=True)
        self.telescopeListFile = self._outputDirectory.joinpath(fileName)

        self._logger.info("Exporting telescope list to ECSV file {}".format(self.telescopeListFile))

        metaData = self._get_export_metadata(crsName == "corsika")

        table = Table(meta=metaData)

        tel_names = list()
        pos_x, pos_y, pos_z = list(), list(), list()
        for tel in self._telescopeList:
            tel_names.append(tel.name)
            x, y, z = tel.getCoordinates(crsName)
            pos_x.append(x)
            pos_y.append(y)
            pos_z.append(z)
        # TODO
        #            try:
        #                pos_z.append(
        #                    tel.convertTelescopeAltitudeToCorsikaSystem(
        #                        z,
        #                        self._corsikaTelescope["corsika_obs_level"],
        #                        self._corsikaTelescope["corsika_sphere_center"][
        #                            self.getTelescopeType(tel.name)
        #                        ],
        #                    )
        #                )
        #            except KeyError:
        #                self._logger.error("Missing definition of CORSIKA sphere center")
        #                raise

        table["telescope_name"] = tel_names

        if len(self._telescopeList) > 0:
            _nameX, _nameY, _nameZ = self._telescopeList[0].getCoordinates(
                crs_name=crsName, coordinate_field="name"
            )
            table[_nameX] = pos_x
            table[_nameY] = pos_y
            table[_nameZ] = pos_z

        table.write(self.telescopeListFile, format="ascii.ecsv", overwrite=True)

    def getNumberOfTelescopes(self):
        """
        Return the number of telescopes in the list.

        Returns
        -------
        int
            Number of telescopes.
        """
        return len(self._telescopeList)

    def getCorsikaInputList(self):
        """
        Get a string with the piece of text to be added to
        the CORSIKA input file.

        Returns
        -------
        str
            Piece of text to be added to the CORSIKA input file.
        """
        corsikaList = ""
        for tel in self._telescopeList:
            posX, posY, posZ = tel.getCoordinates("corsika")
            try:
                sphereRadius = self._corsikaTelescope["corsika_sphere_radius"][
                    self.getTelescopeType(tel.name)
                ]
            except KeyError:
                self._logger.error("Missing definition of CORSIKA sphere radius")
                raise
            try:
                posZ = tel.convertTelescopeAltitudeToCorsikaSystem(
                    posZ,
                    self._corsikaTelescope["corsika_obs_level"],
                    self._corsikaTelescope["corsika_sphere_center"][
                        self.getTelescopeType(tel.name)
                    ],
                )
            except KeyError:
                self._logger.error("Missing definition of CORSIKA sphere center")
                raise

            corsikaList += "TELESCOPE"
            corsikaList += "\t {:.3f}E2".format(posX.value)
            corsikaList += "\t {:.3f}E2".format(posY.value)
            corsikaList += "\t {:.3f}E2".format(posZ.value)
            corsikaList += "\t {:.3f}E2".format(sphereRadius)
            corsikaList += "\t # {}\n".format(tel.name)

        return corsikaList

    def _printAll(self):
        """ "
        Print all columns.

        """

        print("LayoutArray: {}".format(self.name))
        print("ArrayCenter")
        print(self._arrayCenter)
        print("Telescopes")
        for tel in self._telescopeList:
            print(tel)

    def _printCompact(self, compact_printing, corsikaZ):
        """
        Compact printing of list of telescopes


        Parameters
        ----------
        compact_printing: str
            Compact printout for a single coordinate system
        corsikaZ: bool
            Print telescope height in CORSIKA coordinates (for CORSIKA system)

        """

        for tel in self._telescopeList:
            if corsikaZ:
                try:
                    _corsika_obs_level = self._corsikaTelescope["corsika_obs_level"]
                    if self.getTelescopeType(tel.name) is None:
                        _corsika_sphere_center = 0.0 * u.m
                    else:
                        _corsika_sphere_center = self._corsikaTelescope["corsika_sphere_center"][
                            self.getTelescopeType(tel.name)
                        ]
                except KeyError:
                    self._logger.error(
                        "Missing CORSIKA parameters for telescope height calculation"
                    )
                    raise
            else:
                _corsika_obs_level = None
                _corsika_sphere_center = None

            tel.printCompactFormat(
                crs_name=compact_printing,
                print_header=(tel == self._telescopeList[0]),
                corsika_obs_level=_corsika_obs_level,
                corsika_sphere_center=_corsika_sphere_center,
            )

    def printTelescopeList(self, compact_printing="", corsikaZ=False):
        """
        Print list of telescopes in current layout.

        Parameters
        ----------
        compact_printing: str
            Compact printout for a single coordinate system
        corsikaZ: bool
            Print telescope height in CORSIKA coordinates (for CORSIKA system)

        """
        if len(compact_printing) == 0:
            self._printAll()
        else:
            self._printCompact(compact_printing, corsikaZ)

    def convertCoordinates(self):
        """Perform all the possible conversions the coordinates of the tel positions."""

        self._logger.info("Converting telescope coordinates")

        wgs84 = self._getCrsWgs84()
        crs_local = self._getCrsLocal()
        crs_utm = self._getCrsUtm()

        for tel in self._telescopeList:
            tel.convertAll(
                crsLocal=crs_local,
                crsWgs84=wgs84,
                crsUtm=crs_utm,
            )

    def _getCrsLocal(self):
        """
        Local coordinate system definition

        Returns
        -------
        pyproj.CRS
            local coordinate system

        """
        if self._arrayCenter:
            _centerLat, _centerLon, _ = self._arrayCenter.getCoordinates("mercator")
            if not np.isnan(_centerLat.value) and not np.isnan(_centerLon.value):
                proj4_string = (
                    "+proj=tmerc +ellps=WGS84 +datum=WGS84"
                    + " +lon_0={} +lat_0={}".format(_centerLon, _centerLat)
                    + " +axis=nwu +units=m +k_0=1.0"
                )
                crs_local = pyproj.CRS.from_proj4(proj4_string)
                self._logger.debug("Local Mercator projection: {}".format(crs_local))
                return crs_local

        self._logger.debug("crs_local cannot be built: missing array center lon and lat")

    def _getCrsUtm(self):
        """
        UTM coordinate system definition

        Returns
        -------
        pyproj.CRS
            UTM coordinate system

        """
        if self._epsg:
            crs_utm = pyproj.CRS.from_user_input(self._epsg)
            self._logger.debug("UTM system: {}".format(crs_utm))
            return crs_utm
        else:
            self._logger.debug("crs_utm cannot be built because EPSG definition is missing")

    @staticmethod
    def _getCrsWgs84():
        """
        WGS coordinate system definition

        Returns
        -------
        pyproj.CRS
            WGS coordinate system

        """
        return pyproj.CRS("EPSG:4326")

    @staticmethod
    def _layoutCenterDefaults():
        """
        Default values for array center dict

        Returns
        -------
        dict
            array center default values

        """
        return {
            "EPSG": None,
            "center_lon": None,
            "center_lat": None,
            "center_northing": None,
            "center_easting": None,
            "center_alt": None,
        }

    @staticmethod
    def _corsikaTelescopeDefault():
        """
        Default values for CORSIKA telescope dict

        Returns
        -------
        dict
            corsika telescope default values

        TODO: this should go

        """
        return {
            "corsika_obs_level": None,
            "corsika_sphere_center": {
                "LST": 16.0 * u.m,
                "MST": 9.0 * u.m,
                "SST": 3.25 * u.m,
            },
            "corsika_sphere_radius": {
                "LST": 12.5 * u.m,
                "MST": 9.6 * u.m,
                "SST": 3.5 * u.m,
            },
        }

    @staticmethod
    def getTelescopeType(telescope_name):
        """
        Guess telescope type from name

        This does not work consistently, as array elements (non-telescopes)
        might have asset names starting with L,M,S

        TODO: this should go

        """

        _class, _ = names.splitTelescopeModelName(telescope_name)
        try:
            if _class[0:3] in ("LST", "MST", "SST"):
                return _class[0:3]
            if telescope_name[0:2] == "L-":
                return "LST"
            if telescope_name[0:2] == "M-":
                return "MST"
            if telescope_name[0:2] == "S-":
                return "SST"
        except IndexError:
            pass
