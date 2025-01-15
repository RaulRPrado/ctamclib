"""Module to handle interaction with DB."""

import logging
import re
from importlib.resources import files
from pathlib import Path
from threading import Lock

import gridfs
import jsonschema
from bson.objectid import ObjectId
from packaging.version import Version
from pymongo import ASCENDING, MongoClient
from pymongo.errors import BulkWriteError

from simtools.data_model import validate_data
from simtools.db import db_array_elements
from simtools.io_operations import io_handler
from simtools.utils import names, value_conversion

__all__ = ["DatabaseHandler"]

logging.getLogger("pymongo").setLevel(logging.WARNING)


# pylint: disable=unsubscriptable-object
# The above comment is because pylint does not know that DatabaseHandler.db_client is subscriptable


jsonschema_db_dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema#",
    "type": "object",
    "description": "MongoDB configuration",
    "properties": {
        "db_server": {"type": "string", "description": "DB server address"},
        "db_api_port": {
            "type": "integer",
            "minimum": 1,
            "maximum": 65535,
            "default": 27017,
            "description": "Port to use",
        },
        "db_api_user": {"type": "string", "description": "API username"},
        "db_api_pw": {"type": "string", "description": "Password for the API user"},
        "db_api_authentication_database": {
            "type": "string",
            "default": "admin",
            "description": "DB with user info (optional)",
        },
        "db_simulation_model": {
            "type": "string",
            "description": "Name of simulation model database",
        },
    },
    "required": ["db_server", "db_api_port", "db_api_user", "db_api_pw", "db_simulation_model"],
}


class DatabaseHandler:
    """
    DatabaseHandler provides the interface to the DB.

    Parameters
    ----------
    mongo_db_config: dict
        Dictionary with the MongoDB configuration (see jsonschema_db_dict for details).
    """

    DB_CTA_SIMULATION_MODEL_DESCRIPTIONS = "CTA-Simulation-Model-Descriptions"
    # DB collection with updates field names
    DB_DERIVED_VALUES = "Staging-CTA-Simulation-Model-Derived-Values"

    ALLOWED_FILE_EXTENSIONS = [".dat", ".txt", ".lis", ".cfg", ".yml", ".yaml", ".ecsv"]

    db_client = None
    site_parameters_cached = {}
    model_parameters_cached = {}
    model_versions_cached = {}
    corsika_configuration_parameters_cached = {}

    def __init__(self, mongo_db_config=None):
        """Initialize the DatabaseHandler class."""
        self._logger = logging.getLogger(__name__)

        self.mongo_db_config = self._validate_mongo_db_config(mongo_db_config)
        self.io_handler = io_handler.IOHandler()
        self.list_of_collections = {}

        self._set_up_connection()
        self._find_latest_simulation_model_db()

    def _set_up_connection(self):
        """Open the connection to MongoDB."""
        if self.mongo_db_config and DatabaseHandler.db_client is None:
            lock = Lock()
            with lock:
                DatabaseHandler.db_client = self._open_mongo_db()

    def _validate_mongo_db_config(self, mongo_db_config):
        """Validate the MongoDB configuration."""
        if mongo_db_config is None or all(value is None for value in mongo_db_config.values()):
            return None
        try:
            jsonschema.validate(instance=mongo_db_config, schema=jsonschema_db_dict)
            return mongo_db_config
        except jsonschema.exceptions.ValidationError as err:
            raise ValueError("Invalid MongoDB configuration") from err

    def _open_mongo_db(self):
        """
        Open a connection to MongoDB and return the client to read/write to the DB with.

        Returns
        -------
        A PyMongo DB client

        Raises
        ------
        KeyError
            If the DB configuration is invalid
        """
        direct_connection = self.mongo_db_config["db_server"] in (
            "localhost",
            "simtools-mongodb",
        )
        return MongoClient(
            self.mongo_db_config["db_server"],
            port=self.mongo_db_config["db_api_port"],
            username=self.mongo_db_config["db_api_user"],
            password=self.mongo_db_config["db_api_pw"],
            authSource=self.mongo_db_config.get("db_api_authentication_database", "admin"),
            directConnection=direct_connection,
            ssl=not direct_connection,
            tlsallowinvalidhostnames=True,
            tlsallowinvalidcertificates=True,
        )

    def _find_latest_simulation_model_db(self):
        """
        Find the latest released version of the simulation model and update the DB config.

        This is indicated by adding "LATEST" to the name of the simulation model database
        (field "db_simulation_model" in the database configuration dictionary).
        Only released versions are considered, pre-releases are ignored.

        Raises
        ------
        ValueError
            If the "LATEST" version is requested but no versions are found in the DB.

        """
        try:
            if not self.mongo_db_config["db_simulation_model"].endswith("LATEST"):
                return
        except TypeError:
            return

        prefix = self.mongo_db_config["db_simulation_model"].replace("LATEST", "")
        list_of_db_names = self.db_client.list_database_names()
        filtered_list_of_db_names = [s for s in list_of_db_names if s.startswith(prefix)]
        versioned_strings = []
        version_pattern = re.compile(
            rf"{re.escape(prefix)}v?(\d+)-(\d+)-(\d+)(?:-([a-zA-Z0-9_.]+))?"
        )

        for s in filtered_list_of_db_names:
            match = version_pattern.search(s)
            # A version is considered a pre-release if it contains a '-' character (re group 4)
            if match and match.group(4) is None:
                version_str = match.group(1) + "." + match.group(2) + "." + match.group(3)
                version = Version(version_str)
                versioned_strings.append((s, version))

        if versioned_strings:
            latest_string, _ = max(versioned_strings, key=lambda x: x[1])
            self.mongo_db_config["db_simulation_model"] = latest_string
            self._logger.info(
                f"Updated the DB simulation model to the latest version {latest_string}"
            )
        else:
            raise ValueError("Found LATEST in the DB name but no matching versions found in DB.")

    def get_model_parameters(
        self,
        site,
        array_element_name,
        model_version,
        collection="telescope",
        only_applicable=False,
    ):
        """
        Get parameters from MongoDB or simulation model repository for an array element.

        An array element can be e.g., a telescope or a calibration device.
        Read parameters for design and for the specified array element (if necessary). This allows
        to overwrite design parameters with specific parameters without having to copy
        all model parameters when changing only a few.

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. LSTN-01, MSTS-design)
        model_version: str
            Version of the model.
        collection: str
            collection of array element (e.g. telescopes, calibration_devices)
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        """
        _site, _array_element_name, _model_version = self._validate_model_input(
            site, array_element_name, model_version
        )
        array_element_list = db_array_elements.get_array_element_list_for_db_query(
            _array_element_name, self, _model_version, collection
        )
        pars = {}
        for array_element in array_element_list:
            _array_elements_cache_key = self._parameter_cache_key(
                site, array_element, model_version, collection
            )
            try:
                pars.update(DatabaseHandler.model_parameters_cached[_array_elements_cache_key])
            except KeyError:
                pars.update(
                    self.read_mongo_db(
                        self.mongo_db_config.get("db_simulation_model", None),
                        array_element_name=array_element,
                        model_version=_model_version,
                        collection_name=collection,
                        run_location=None,
                        write_files=False,
                        only_applicable=only_applicable,
                    )
                )
            DatabaseHandler.model_parameters_cached[_array_elements_cache_key] = pars

        return pars

    def get_collection(self, db_name, collection_name):
        """
        Get a collection from the DB.

        Parameters
        ----------
        db_name: str
            Name of the DB.
        collection_name: str
            Name of the collection.

        Returns
        -------
        pymongo.collection.Collection
            The collection from the DB.

        """
        db_name = self._get_db_name(db_name)
        return DatabaseHandler.db_client[db_name][collection_name]

    def export_file_db(self, db_name, dest, file_name):
        """
        Get file from the DB and write to disk.

        Parameters
        ----------
        db_name: str
            Name of the DB to search in.
        dest: str or Path
            Location where to write the file to.
        file_name: str
            Name of the file to get.

        Returns
        -------
        file_id: GridOut._id
            the database ID the file was assigned when it was inserted to the DB.

        Raises
        ------
        FileNotFoundError
            If the desired file is not found.

        """
        db_name = self._get_db_name(db_name)

        self._logger.debug(f"Getting {file_name} from {db_name} and writing it to {dest}")
        file_path_instance = self._get_file_mongo_db(db_name, file_name)
        self._write_file_from_mongo_to_disk(db_name, dest, file_path_instance)
        return file_path_instance._id  # pylint: disable=protected-access;

    def export_model_files(self, parameters, dest):
        """
        Export all the files in a model from the DB and write them to disk.

        Parameters
        ----------
        parameters: dict
            Dict of model parameters.
        dest: str or Path
            Location where to write the files to.

        Raises
        ------
        FileNotFoundError
            if a file in parameters.values is not found

        """
        if self.mongo_db_config:
            for info in parameters.values():
                if not info or not info.get("file") or info["value"] is None:
                    continue
                if Path(dest).joinpath(info["value"]).exists():
                    continue
                file = self._get_file_mongo_db(self._get_db_name(), info["value"])
                self._write_file_from_mongo_to_disk(self._get_db_name(), dest, file)

    @staticmethod
    def _is_file(value):
        """Verify if a parameter value is a file name."""
        return any(ext in str(value) for ext in DatabaseHandler.ALLOWED_FILE_EXTENSIONS)

    def read_mongo_db(
        self,
        db_name,
        array_element_name,
        model_version,
        run_location,
        collection_name,
        write_files=True,
        only_applicable=False,
    ):
        """
        Build and execute query to Read the MongoDB for a specific array element.

        Also writes the files listed in the parameter values into the sim_telarray run location

        Parameters
        ----------
        db_name: str
            the name of the DB
        array_element_name: str
            Name of the array element model (e.g. MSTN-design ...)
        model_version: str
            Version of the model.
        run_location: Path or str
            The sim_telarray run location to write the tabulated data files into.
        collection_name: str
            The name of the collection to read from.
        write_files: bool
            If true, write the files to the run_location.
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        Raises
        ------
        ValueError
            if query returned zero results.

        """
        collection = self.get_collection(db_name, collection_name)
        _parameters = {}

        query = {
            "instrument": array_element_name,
            "version": self.model_version(model_version, db_name),
        }

        if only_applicable:
            query["applicable"] = True
        if collection.count_documents(query) < 1:
            raise ValueError(
                "The following query returned zero results! Check the input data and rerun.\n",
                query,
            )
        for post in collection.find(query).sort("parameter", ASCENDING):
            par_now = post["parameter"]
            _parameters[par_now] = post
            _parameters[par_now].pop("parameter", None)
            _parameters[par_now].pop("instrument", None)
            _parameters[par_now]["entry_date"] = ObjectId(post["_id"]).generation_time
            if _parameters[par_now]["file"] and write_files:
                file = self._get_file_mongo_db(db_name, _parameters[par_now]["value"])
                self._write_file_from_mongo_to_disk(db_name, run_location, file)

        return _parameters

    def get_site_parameters(
        self,
        site,
        model_version,
        only_applicable=False,
    ):
        """
        Get parameters from either MongoDB or simulation model repository for a specific site.

        Parameters
        ----------
        site: str
            Site name.
        model_version: str
            Version of the model.
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        """
        _site, _, _model_version = self._validate_model_input(site, None, model_version)
        _db_name = self._get_db_name()
        _site_cache_key = self._parameter_cache_key(site, None, model_version)
        try:
            return DatabaseHandler.site_parameters_cached[_site_cache_key]
        except KeyError:
            pass

        DatabaseHandler.site_parameters_cached[_site_cache_key] = (
            self._get_site_parameters_mongo_db(
                _db_name,
                _site,
                _model_version,
                only_applicable,
            )
        )
        return DatabaseHandler.site_parameters_cached[_site_cache_key]

    def _get_site_parameters_mongo_db(self, db_name, site, model_version, only_applicable=False):
        """
        Get parameters from MongoDB for a specific site.

        Parameters
        ----------
        db_name: str
            The name of the DB.
        site: str
            Site name.
        model_version: str
            Version of the model.
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        Raises
        ------
        ValueError
            if query returned zero results.

        """
        collection = self.get_collection(db_name, "sites")
        _parameters = {}

        query = {
            "site": site,
            "version": model_version,
        }
        if only_applicable:
            query["applicable"] = True
        if collection.count_documents(query) < 1:
            raise ValueError(
                "The following query returned zero results! Check the input data and rerun.\n",
                query,
            )
        for post in collection.find(query).sort("parameter", ASCENDING):
            par_now = post["parameter"]
            _parameters[par_now] = post
            _parameters[par_now].pop("parameter", None)
            _parameters[par_now].pop("site", None)
            _parameters[par_now]["entry_date"] = ObjectId(post["_id"]).generation_time

        return _parameters

    def get_derived_values(self, site, array_element_name, model_version):
        """
        Get all derived values from the DB for a specific array element.

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. MSTN, SSTS).
        model_version: str
            Version of the model.

        Returns
        -------
        dict containing the parameters

        """
        _, _array_element_name, _model_version = self._validate_model_input(
            site, array_element_name, model_version
        )

        return self.read_mongo_db(
            DatabaseHandler.DB_DERIVED_VALUES,
            _array_element_name,
            _model_version,
            run_location=None,
            collection_name="derived_values",
            write_files=False,
        )

    def get_simulation_configuration_parameters(
        self, simulation_software, site, array_element_name, model_version
    ):
        """
        Get simulation configuration parameters from the DB.

        Parameters
        ----------
        simulation_software: str
            Name of the simulation software.
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. MSTN, SSTS).
        model_version: str
            Version of the model.

        Returns
        -------
        dict containing the parameters

        Raises
        ------
        ValueError
            if simulation_software is not valid.
        """
        if simulation_software == "corsika":
            return self.get_corsika_configuration_parameters(model_version)
        if simulation_software == "simtel":
            if site and array_element_name:
                return self.get_model_parameters(
                    site, array_element_name, model_version, collection="configuration_sim_telarray"
                )
            return {}
        raise ValueError(f"Unknown simulation software: {simulation_software}")

    def get_corsika_configuration_parameters(self, model_version):
        """
        Get CORSIKA configuration parameters from the DB.

        Parameters
        ----------
        model_version : str
            Version of the model.

        Returns
        -------
        dict
            Configuration parameters for CORSIKA
        """
        _corsika_cache_key = self._parameter_cache_key(None, None, model_version)
        try:
            return DatabaseHandler.corsika_configuration_parameters_cached[_corsika_cache_key]
        except KeyError:
            pass
        DatabaseHandler.corsika_configuration_parameters_cached[_corsika_cache_key] = (
            self.read_mongo_db(
                db_name=self._get_db_name(),
                array_element_name=None,
                model_version=model_version,
                run_location=None,
                collection_name="configuration_corsika",
                write_files=False,
            )
        )
        return DatabaseHandler.corsika_configuration_parameters_cached[_corsika_cache_key]

    def _validate_model_input(self, site, array_element_name, model_version):
        """
        Validate input for model parameter queries.

        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. LSTN-01, MSTS-design)
        model_version: str
            Version of the model.

        """
        return (
            names.validate_site_name(site),
            names.validate_array_element_name(array_element_name) if array_element_name else None,
            self.model_version(model_version),
        )

    @staticmethod
    def _get_file_mongo_db(db_name, file_name):
        """
        Extract a file from MongoDB and return GridFS file instance.

        Parameters
        ----------
        db_name: str
            the name of the DB with files of tabulated data
        file_name: str
            The name of the file requested

        Returns
        -------
        GridOut
            A file instance returned by GridFS find_one

        Raises
        ------
        FileNotFoundError
            If the desired file is not found.

        """
        db = DatabaseHandler.db_client[db_name]
        file_system = gridfs.GridFS(db)
        if file_system.exists({"filename": file_name}):
            return file_system.find_one({"filename": file_name})

        raise FileNotFoundError(f"The file {file_name} does not exist in the database {db_name}")

    @staticmethod
    def _write_file_from_mongo_to_disk(db_name, path, file):
        """
        Extract a file from MongoDB and write it to disk.

        Parameters
        ----------
        db_name: str
            the name of the DB with files of tabulated data
        path: str or Path
            The path to write the file to
        file: GridOut
            A file instance returned by GridFS find_one
        """
        db = DatabaseHandler.db_client[db_name]
        fs_output = gridfs.GridFSBucket(db)
        with open(Path(path).joinpath(file.filename), "wb") as output_file:
            fs_output.download_to_stream_by_name(file.filename, output_file)

    def copy_array_element(
        self,
        db_name,
        element_to_copy,
        version_to_copy,
        new_array_element_name,
        collection_name="telescopes",
        db_to_copy_to=None,
        collection_to_copy_to=None,
    ):
        """
        Copy a full array element configuration to a new array element name.

        Only a specific version is copied.
        This function should be rarely used and is intended to simplify unit tests.

        Parameters
        ----------
        db_name: str
            the name of the DB to copy from
        element_to_copy: str
            The array element to copy
        version_to_copy: str
            The version of the configuration to copy
        new_array_element_name: str
            The name of the new array element
        collection_name: str
            The name of the collection to copy from.
        db_to_copy_to: str
            The name of the DB to copy to.
        collection_to_copy_to: str
            The name of the collection to copy to.

        Raises
        ------
        BulkWriteError

        """
        db_name = self._get_db_name(db_name)
        if db_to_copy_to is None:
            db_to_copy_to = db_name

        if collection_to_copy_to is None:
            collection_to_copy_to = collection_name

        self._logger.info(
            f"Copying version {version_to_copy} of {element_to_copy} "
            f"to the new array element {new_array_element_name} in the {db_to_copy_to} DB"
        )

        collection = self.get_collection(db_name, collection_name)
        db_entries = []

        _version_to_copy = self.model_version(version_to_copy)

        query = {
            "instrument": element_to_copy,
            "version": _version_to_copy,
        }
        for post in collection.find(query):
            post["instrument"] = new_array_element_name
            post.pop("_id", None)
            db_entries.append(post)

        self._logger.info(f"Creating new array element {new_array_element_name}")
        collection = self.get_collection(db_to_copy_to, collection_to_copy_to)
        try:
            collection.insert_many(db_entries)
        except BulkWriteError as exc:
            raise BulkWriteError(str(exc.details)) from exc

    def add_new_parameter(
        self,
        db_name,
        par_dict,
        collection_name="telescopes",
        file_prefix=None,
    ):
        """
        Add a parameter dictionary for a specific array element to the DB.

        A new document will be added to the DB, with all fields taken from the input parameters.
        Parameter dictionaries are validated before submission using the corresponding schema.

        Parameters
        ----------
        db_name: str
            the name of the DB
        par_dict: dict
            dictionary with parameter data
        collection_name: str
            The name of the collection to add a parameter to.
        file_prefix: str or Path
            where to find files to upload to the DB
        """
        data_validator = validate_data.DataValidator(
            schema_file=files("simtools")
            / f"schemas/model_parameters/{par_dict['parameter']}.schema.yml",
            data_dict=par_dict,
            check_exact_data_type=False,
        )
        par_dict = data_validator.validate_and_transform(is_model_parameter=True)

        db_name = self._get_db_name(db_name)
        collection = self.get_collection(db_name, collection_name)

        par_dict["value"], _base_unit, _ = value_conversion.get_value_unit_type(
            value=par_dict["value"], unit_str=par_dict.get("unit", None)
        )
        par_dict["unit"] = _base_unit if _base_unit else None

        files_to_add_to_db = set()
        if par_dict["file"] and par_dict["value"]:
            if file_prefix is None:
                raise FileNotFoundError(
                    "The location of the file to upload, "
                    f"corresponding to the {par_dict['parameter']} parameter, must be provided."
                )
            file_path = Path(file_prefix).joinpath(par_dict["value"])
            files_to_add_to_db.add(f"{file_path}")

        self._logger.info(
            f"Adding a new entry to DB {db_name} and collection {db_name}:\n{par_dict}"
        )
        collection.insert_one(par_dict)

        for file_to_insert_now in files_to_add_to_db:
            self._logger.info(f"Will also add the file {file_to_insert_now} to the DB")
            self.insert_file_to_db(file_to_insert_now, db_name)

        self._reset_parameter_cache(par_dict["site"], par_dict["instrument"], par_dict["version"])

    def _get_db_name(self, db_name=None):
        """
        Return database name. If not provided, return the default database name.

        Parameters
        ----------
        db_name: str
            Database name

        Returns
        -------
        str
            Database name
        """
        return self.mongo_db_config["db_simulation_model"] if db_name is None else db_name

    def model_version(self, version, db_name=None):
        """
        Return model version and check that it is valid.

        Queries the database for all available model versions and check if the
        requested version is valid.

        Parameters
        ----------
        version : str
            Model version.
        db_name : str
            Database name.

        Returns
        -------
        str
            Model version.

        Raises
        ------
        ValueError
            if version not valid.

        """
        _all_versions = self.get_all_versions(db_name=db_name)
        if version in _all_versions:
            return version
        if len(_all_versions) == 0:
            return None

        raise ValueError(
            f"Invalid model version {version} in DB {self._get_db_name(db_name)} "
            f"(allowed are {_all_versions})"
        )

    def insert_file_to_db(self, file_name, db_name=None, **kwargs):
        """
        Insert a file to the DB.

        Parameters
        ----------
        file_name: str or Path
            The name of the file to insert (full path).
        db_name: str
            the name of the DB
        **kwargs (optional): keyword arguments for file creation.
            The full list of arguments can be found in, \
            https://docs.mongodb.com/manual/core/gridfs/#the-files-collection
            mostly these are unnecessary though.

        Returns
        -------
        file_iD: GridOut._id
            If the file exists, return its GridOut._id, otherwise insert the file and return its"
            "newly created DB GridOut._id.

        """
        db_name = self._get_db_name(db_name)

        db = DatabaseHandler.db_client[db_name]
        file_system = gridfs.GridFS(db)

        if "content_type" not in kwargs:
            kwargs["content_type"] = "ascii/dat"

        if "filename" not in kwargs:
            kwargs["filename"] = Path(file_name).name

        if file_system.exists({"filename": kwargs["filename"]}):
            self._logger.warning(
                f"The file {kwargs['filename']} exists in the DB. Returning its ID"
            )
            return file_system.find_one(  # pylint: disable=protected-access
                {"filename": kwargs["filename"]}
            )._id
        with open(file_name, "rb") as data_file:
            return file_system.put(data_file, **kwargs)

    def get_all_versions(
        self,
        parameter=None,
        array_element_name=None,
        site=None,
        db_name=None,
        collection=None,
    ):
        """
        Get all version entries in the DB of collection and/or a specific parameter.

        Parameters
        ----------
        parameter: str
            Which parameter to get the versions of
        array_element_name: str
            Which array element to get the versions of (in case "collection_name" is not "sites")
        site: str
            Site name.
        db_name: str
            Database name.
        collection_name: str
            The name of the collection in which to update the parameter.

        Returns
        -------
        all_versions: list
            List of all versions found

        Raises
        ------
        ValueError
            If key to collection_name is not valid.

        """
        db_name = self._get_db_name() if db_name is None else db_name
        if not db_name:
            self._logger.warning("No database name defined to determine list of model versions")
            return []
        _cache_key = f"model_versions_{db_name}-{collection}"

        query = {}
        if parameter is not None:
            query["parameter"] = parameter
            _cache_key = f"{_cache_key}-{parameter}"
        if collection in ["telescopes", "calibration_devices"] and array_element_name is not None:
            query["instrument"] = names.validate_array_element_name(array_element_name)
            _cache_key = f"{_cache_key}-{query['instrument']}"
        elif collection == "sites" and site is not None:
            query["site"] = names.validate_site_name(site)
            _cache_key = f"{_cache_key}-{query['site']}"

        if _cache_key not in DatabaseHandler.model_versions_cached:
            all_versions = set()
            collections_to_query = (
                [collection] if collection else self.get_collections(db_name, True)
            )
            for collection_name in collections_to_query:
                db_collection = self.get_collection(db_name, collection_name)
                sorted_posts = db_collection.find(query).sort("version", ASCENDING)
                all_versions.update(post["version"] for post in sorted_posts)
            DatabaseHandler.model_versions_cached[_cache_key] = list(all_versions)

        if len(DatabaseHandler.model_versions_cached[_cache_key]) == 0:
            self._logger.warning(f"The query {query} did not return any results. No versions found")

        return DatabaseHandler.model_versions_cached[_cache_key]

    def _parameter_cache_key(self, site, array_element_name, model_version, collection=None):
        """
        Create a cache key for the parameter cache dictionaries.

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Array element name.
        model_version: str
            Model version.
        collection: str
            DB collection name.

        Returns
        -------
        str
            Cache key.
        """
        parts = []
        if site:
            parts.append(site)
        if array_element_name:
            parts.append(array_element_name)
        parts.append(model_version)
        if collection:
            parts.append(collection)
        return "-".join(parts)

    def _reset_parameter_cache(self, site, array_element_name, model_version):
        """
        Reset the cache for the parameters.

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Array element name.
        model_version: str
            Model version.
        """
        self._logger.debug(f"Resetting cache for {site} {array_element_name} {model_version}")
        _cache_key = self._parameter_cache_key(site, array_element_name, model_version)
        DatabaseHandler.site_parameters_cached.pop(_cache_key, None)
        DatabaseHandler.model_parameters_cached.pop(_cache_key, None)
        db_array_elements.get_array_elements.cache_clear()

    def get_collections(self, db_name=None, model_collections_only=False):
        """
        List of collections in the DB.

        Parameters
        ----------
        db_name: str
            Database name.

        Returns
        -------
        list
            List of collection names
        model_collections_only: bool
            If True, only return model collections (i.e. exclude fs.files, fs.chunks, metadata)

        """
        db_name = self._get_db_name() if db_name is None else db_name
        if db_name not in self.list_of_collections:
            self.list_of_collections[db_name] = DatabaseHandler.db_client[
                db_name
            ].list_collection_names()
        if model_collections_only:
            return [
                collection
                for collection in self.list_of_collections[db_name]
                if not collection.startswith("fs.") and collection != "metadata"
            ]
        return self.list_of_collections[db_name]
