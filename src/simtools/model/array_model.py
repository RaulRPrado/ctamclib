"""Definition of the ArrayModel class."""

import logging
from pathlib import Path

import astropy.units as u
from astropy.table import QTable

from simtools.data_model import data_reader, schema
from simtools.db import db_handler
from simtools.io_operations import io_handler
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.utils import general, names

__all__ = ["ArrayModel"]


class ArrayModel:
    """
    Representation of an observatory consisting of site, telescopes, and further devices.

    Parameters
    ----------
    mongo_db_config: dict
        MongoDB configuration.
    model_version: str
        Model version.
    label: str, optional
        Instance label. Used for output file naming.
    site: str, optional
        Site name.
    layout_name: str, optional
        Layout name.
    array_elements: Union[str, Path, List[str]], optional
        Array element definitions (list of array element or path to file with
        the array element positions).
    """

    def __init__(
        self,
        mongo_db_config: dict,
        model_version: str,
        label: str | None = None,
        site: str | None = None,
        layout_name: str | None = None,
        array_elements: str | Path | list[str] | None = None,
    ):
        """Initialize ArrayModel."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init ArrayModel")
        self.mongo_db_config = mongo_db_config
        self.model_version = model_version
        self.label = label
        self.layout_name = layout_name
        self._config_file_path = None
        self._config_file_directory = None
        self.io_handler = io_handler.IOHandler()
        self.db = db_handler.DatabaseHandler(mongo_db_config=mongo_db_config)

        self.array_elements, self.site_model, self.telescope_model = self._initialize(
            site, array_elements
        )

        self._telescope_model_files_exported = False
        self._array_model_file_exported = False

    def _initialize(self, site: str, array_elements_config: str | Path | list[str]):
        """
        Initialize ArrayModel taking different configuration options into account.

        Parameters
        ----------
        site: str
            Site name.
        array_elements_config: Union[str, Path, List[str]]
            Array element definitions.

        Returns
        -------
        dict
            Dict with telescope positions.
        SiteModel
            Site model.
        dict
            Dict with telescope models.
        """
        self._logger.debug(f"Getting site parameters from DB ({site})")
        site_model = SiteModel(
            site=names.validate_site_name(site),
            mongo_db_config=self.mongo_db_config,
            model_version=self.model_version,
            label=self.label,
        )

        array_elements = {}
        # Case 1: array_elements is a file name
        if isinstance(array_elements_config, str | Path):
            array_elements = self._load_array_element_positions_from_file(
                array_elements_config, site
            )
        # Case 2: array elements is a list of elements
        elif isinstance(array_elements_config, list):
            array_elements = self._get_array_elements_from_list(array_elements_config)
        # Case 3: array elements defined in DB by array layout name
        elif self.layout_name is not None:
            array_elements = self._get_array_elements_from_list(
                site_model.get_array_elements_for_layout(self.layout_name)
            )
        if not array_elements:
            raise ValueError(
                "No array elements found. "
                "Possibly missing valid layout name or missing telescope list."
            )
        telescope_model = self._build_telescope_models(site_model, array_elements)
        return array_elements, site_model, telescope_model

    @property
    def number_of_telescopes(self) -> int:
        """
        Return the number of telescopes.

        Returns
        -------
        int
            Number of telescopes.
        """
        return len(self.telescope_model)

    @property
    def site(self) -> str:
        """
        Return site.

        Returns
        -------
        str
            Site name.
        """
        return self.site_model.site

    def _build_telescope_models(self, site_model: SiteModel, array_elements: dict) -> dict:
        """
        Build the the telescope models for all telescopes of this array.

        Includes reading of telescope model parameters from the DB.
        The array is defined in the telescopes dictionary. Array element positions
        are read from the database if no values are given in this dictionary.

        Parameters
        ----------
        site_model: SiteModel
            Site model.
        array_elements: dict
            Dict with array elements.

        Returns
        -------
        dict
            Dictionary with telescope models.
        """
        telescope_model = {}
        for element_name, _ in array_elements.items():
            collection = names.get_collection_name_from_array_element_name(element_name)
            if collection == "telescopes":
                telescope_model[element_name] = TelescopeModel(
                    site=site_model.site,
                    telescope_name=element_name,
                    model_version=self.model_version,
                    mongo_db_config=self.mongo_db_config,
                    label=self.label,
                )
        return telescope_model

    def print_telescope_list(self):
        """Print list of telescopes."""
        for tel_name, data in self.telescope_model.items():
            print(f"Name: {tel_name}\t Model: {data.name}")

    def export_simtel_telescope_config_files(self):
        """Export sim_telarray configuration files for all telescopes into the model directory."""
        exported_models = []
        for _, tel_model in self.telescope_model.items():
            name = tel_model.name + (
                "_" + tel_model.extra_label if tel_model.extra_label != "" else ""
            )
            if name not in exported_models:
                self._logger.debug(f"Exporting configuration file for telescope {name}")
                tel_model.export_config_file()
                exported_models.append(name)
            else:
                self._logger.debug(
                    f"Configuration file for telescope {name} already exists - skipping"
                )

        self._telescope_model_files_exported = True

    def export_simtel_array_config_file(self):
        """Export sim_telarray configuration file for the array into the model directory."""
        # Setting file name and the location
        config_file_name = names.simtel_config_file_name(
            array_name=self.layout_name,
            site=self.site_model.site,
            model_version=self.model_version,
            label=self.label,
        )
        self._config_file_path = self.get_config_directory().joinpath(config_file_name)

        # Writing parameters to the file
        self._logger.info(f"Writing array configuration file into {self._config_file_path}")
        simtel_writer = SimtelConfigWriter(
            site=self.site_model.site,
            layout_name=self.layout_name,
            model_version=self.model_version,
            label=self.label,
        )
        simtel_writer.write_array_config_file(
            config_file_path=self._config_file_path,
            telescope_model=self.telescope_model,
            site_model=self.site_model,
        )
        self._array_model_file_exported = True

    def export_all_simtel_config_files(self):
        """
        Export sim_telarray config file for the array and for each individual telescope.

        Config files are exported into the output model directory.
        """
        if not self._telescope_model_files_exported:
            self.export_simtel_telescope_config_files()
        if not self._array_model_file_exported:
            self.export_simtel_array_config_file()

    def get_config_file(self) -> Path:
        """
        Return the path of the array config file for sim_telarray.

        A new config file is produced if the file is not updated.

        Returns
        -------
        Path
            Path of the exported config file for sim_telarray.
        """
        self.export_all_simtel_config_files()
        return self._config_file_path

    def get_config_directory(self) -> Path:
        """
        Get the path of the array config directory for sim_telarray.

        Returns
        -------
        Path
            Path of the config directory path for sim_telarray.
        """
        if self._config_file_directory is None:
            self._config_file_directory = self.io_handler.get_output_directory(self.label, "model")
        return self._config_file_directory

    def _load_array_element_positions_from_file(
        self, array_elements_file: str | Path, site: str
    ) -> dict:
        """
        Load array element (e.g. telescope) positions from a file into a dict.

        Dictionary format: {telescope_name: {position_x: x, position_y: y, position_z: z}}

        Parameters
        ----------
        array_elements_file: Union[str, Path]
            Path to the file with the array element positions.
        site: str
            Site name.

        Returns
        -------
        dict
            Dict with telescope positions.
        """
        table = data_reader.read_table_from_file(file_name=array_elements_file)

        return {
            row["telescope_name"]: self._get_telescope_position_parameter(
                row["telescope_name"], site, row["position_x"], row["position_y"], row["position_z"]
            )
            for row in table
        }

    def _get_telescope_position_parameter(
        self,
        telescope_name: str,
        site: str,
        x: u.Quantity,
        y: u.Quantity,
        z: u.Quantity,
        parameter_version: str | None = None,
    ) -> dict:
        """
        Return dictionary with telescope position parameters (following DB model database format).

        Parameters
        ----------
        telescope_name: str
            Name of the telescope.
        site: str
            Site name.
        x: astropy.Quantity
            X ground position.
        y: astropy.Quantity
            Y ground position.
        z: astropy.Quantity
            Z ground position.

        Returns
        -------
        dict
            Dict with telescope position parameters.
        """
        return {
            "schema_version": schema.model_parameter_schema_version(),
            "parameter": "array_element_position_ground",
            "instrument": telescope_name,
            "site": site,
            "parameter_version": parameter_version,
            "unique_id": None,
            "value": general.convert_list_to_string(
                [x.to("m").value, y.to("m").value, z.to("m").value]
            ),
            "unit": "m",
            "type": "float64",
            "file": False,
        }

    def _get_array_elements_from_list(self, array_elements_list: list[str]) -> dict:
        """
        Return dictionary with array elements from a list of telescope names.

        Input list can contain telescope names (e.g, LSTN-01) or a telescope
        type (e.g., MSTN). In the latter case, all telescopes of this specific
        type are added.

        Parameters
        ----------
        array_elements_list: list
            List of telescope names.

        Returns
        -------
        dict
            Dict with array elements.
        """
        array_elements_dict = {}
        for name in array_elements_list:
            try:
                array_elements_dict[names.validate_array_element_name(name)] = None
            except ValueError:
                array_elements_dict.update(self._get_all_array_elements_of_type(name))
        return array_elements_dict

    def _get_all_array_elements_of_type(self, array_element_type: str) -> dict:
        """
        Return all array elements of a specific type using the database.

        Parameters
        ----------
        array_element_type : str
            Type of the array element (e.g. LSTN, MSTS)

        Returns
        -------
        dict
            Dict with array elements.
        """
        all_elements = self.db.get_array_elements_of_type(
            array_element_type=array_element_type,
            model_version=self.model_version,
            collection="telescopes",
        )
        return self._get_array_elements_from_list(all_elements)

    def export_array_elements_as_table(self, coordinate_system: str = "ground") -> QTable:
        """
        Export array elements positions to astropy table.

        Parameters
        ----------
        coordinate_system: str
            Positions are exported in this coordinate system.

        Returns
        -------
        astropy.table.QTable
            Astropy table with the telescope layout information.
        """
        table = QTable(meta={"array_name": self.layout_name, "site": self.site_model.site})

        name, pos_x, pos_y, pos_z, tel_r = [], [], [], [], []
        for tel_name, data in self.telescope_model.items():
            name.append(tel_name)
            xyz = data.position(coordinate_system=coordinate_system)
            pos_x.append(xyz[0])
            pos_y.append(xyz[1])
            pos_z.append(xyz[2])
            try:
                # add tests of KeyError after positions calibration_elements are added to DB
                tel_r.append(data.get_parameter_value_with_unit("telescope_sphere_radius"))
            except KeyError:  # not all array elements have a sphere radius
                tel_r.append(0.0 * u.m)

        table["telescope_name"] = name
        if coordinate_system == "ground":
            table["position_x"] = pos_x
            table["position_y"] = pos_y
            table["position_z"] = pos_z
        elif coordinate_system == "utm":
            table["utm_east"] = pos_x
            table["utm_north"] = pos_y
            table["altitude"] = pos_z
        table["sphere_radius"] = tel_r

        table.sort("telescope_name")
        return table
