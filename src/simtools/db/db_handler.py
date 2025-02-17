"""Module to handle interaction with DB."""

import logging
import re
from pathlib import Path
from threading import Lock

import gridfs
import jsonschema
from bson.objectid import ObjectId
from packaging.version import Version
from pymongo import MongoClient

from simtools.data_model import validate_data
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

    ALLOWED_FILE_EXTENSIONS = [".dat", ".txt", ".lis", ".cfg", ".yml", ".yaml", ".ecsv"]

    db_client = None
    production_table_cached = {}
    model_parameters_cached = {}

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
            db_simulation_model = self.mongo_db_config["db_simulation_model"]
            if not db_simulation_model.endswith("LATEST"):
                return
        except TypeError:  # db_simulation_model is None
            return

        prefix = db_simulation_model.replace("LATEST", "")
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

    def get_model_parameter(
        self,
        parameter,
        parameter_version,
        site,
        array_element_name,
        collection,
    ):
        """
        Get a model parameter using the parameter version.

        Parameters
        ----------
        parameter: str
            Name of the parameter.
        parameter_version: str
            Version of the parameter.
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. MSTN, SSTS).
        collection: str
            Collection of array element (e.g. telescopes, calibration_devices).

        Returns
        -------
        dict containing the parameter

        """
        query = {
            "parameter_version": parameter_version,
            "parameter": parameter,
        }
        if array_element_name is not None:
            query["instrument"] = array_element_name
        if site is not None:
            query["site"] = site
        return self._read_mongo_db(query=query, collection_name=collection)

    def get_model_parameters(
        self,
        site,
        array_element_name,
        collection,
        model_version=None,
    ):
        """
        Get model parameters using the model version.

        Queries parameters for design and for the specified array element (if necessary).

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. LSTN-01, MSTx-FlashCam, ILLN-01).
        model_version: str, list
            Version(s) of the model.
        collection: str
            Collection of array element (e.g. telescopes, calibration_devices).

        Returns
        -------
        dict containing the parameters
        """
        model_versions = (
            self.get_model_versions(collection) if model_version is None else [model_version]
        )

        pars = {}
        for _model_version in model_versions:
            production_table = self._read_production_table_from_mongo_db(collection, _model_version)
            array_element_list = self._get_array_element_list(
                array_element_name, site, production_table, collection
            )
            for array_element in array_element_list:
                pars.update(
                    self._get_parameter_for_model_version(
                        array_element, _model_version, site, collection, production_table
                    )
                )
        return pars

    def _get_parameter_for_model_version(
        self, array_element, model_version, site, collection, production_table
    ):
        cache_key, cache_dict = self._read_cache(
            DatabaseHandler.model_parameters_cached,
            names.validate_site_name(site) if site else None,
            array_element,
            model_version,
            collection,
        )
        if cache_dict:
            self._logger.debug(f"Found {array_element} in cache (key: {cache_key})")
            return cache_dict
        self._logger.debug(f"Did not find {array_element} in cache (key: {cache_key})")

        try:
            parameter_version_table = production_table["parameters"][array_element]
        except KeyError:  # allow missing array elements (parameter dict is checked later)
            return {}
        DatabaseHandler.model_parameters_cached[cache_key] = self._read_mongo_db(
            query=self._get_query_from_parameter_version_table(
                parameter_version_table, array_element, site
            ),
            collection_name=collection,
        )
        return DatabaseHandler.model_parameters_cached[cache_key]

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

    def get_collections(self, db_name=None, model_collections_only=False):
        """
        List of collections in the DB.

        Parameters
        ----------
        db_name: str
            Database name.
        model_collections_only: bool
            If True, only return model collections (i.e. exclude fs.files, fs.chunks)

        Returns
        -------
        list
            List of collection names

        """
        db_name = db_name or self._get_db_name()
        if db_name not in self.list_of_collections:
            self.list_of_collections[db_name] = DatabaseHandler.db_client[
                db_name
            ].list_collection_names()
        collections = self.list_of_collections[db_name]
        if model_collections_only:
            return [collection for collection in collections if not collection.startswith("fs.")]
        return collections

    def export_model_files(self, parameters=None, file_names=None, dest=None, db_name=None):
        """
        Export files from the DB to the model directory.

        The files to be exported can be specified by file_name or retrieved from the database
        using the parameters dictionary.

        Parameters
        ----------
        parameters: dict
            Dict of model parameters
        file_names: list, str
            List (or string) of file names to export
        dest: str or Path
            Location where to write the files to.

        Returns
        -------
        file_id: dict of GridOut._id
            Dict of database IDs of files.
        """
        db_name = self._get_db_name(db_name)

        if file_names:
            file_names = [file_names] if not isinstance(file_names, list) else file_names
        elif parameters:
            file_names = [
                info["value"]
                for info in parameters.values()
                if info and info.get("file") and info["value"] is not None
            ]

        instance_ids = {}
        for file_name in file_names:
            if Path(dest).joinpath(file_name).exists():
                instance_ids[file_name] = "file exists"
            else:
                file_path_instance = self._get_file_mongo_db(self._get_db_name(), file_name)
                self._write_file_from_mongo_to_disk(self._get_db_name(), dest, file_path_instance)
                instance_ids[file_name] = file_path_instance._id  # pylint: disable=protected-access
        return instance_ids

    def _get_query_from_parameter_version_table(
        self, parameter_version_table, array_element_name, site
    ):
        """Return query based on parameter version table."""
        query_dict = {
            "$or": [
                {"parameter": param, "parameter_version": version}
                for param, version in parameter_version_table.items()
            ],
        }
        # 'xSTX-design' is a placeholder to ignore 'instrument' field in query.
        if array_element_name and array_element_name != "xSTx-design":
            query_dict["instrument"] = array_element_name
        if site:
            query_dict["site"] = site
        return query_dict

    def _read_mongo_db(self, query, collection_name):
        """
        Query MongoDB.

        Parameters
        ----------
        query: dict
            Query to execute.
        collection_name: str
            Collection name.

        Returns
        -------
        dict containing the parameters

        Raises
        ------
        ValueError
            if query returned no results.
        """
        db_name = self._get_db_name()
        collection = self.get_collection(db_name, collection_name)
        posts = list(collection.find(query))
        if not posts:
            raise ValueError(
                f"The following query for {collection_name} returned zero results: {query} "
            )
        parameters = {}
        for post in posts:
            par_now = post["parameter"]
            parameters[par_now] = post
            parameters[par_now]["entry_date"] = ObjectId(post["_id"]).generation_time
        return {k: parameters[k] for k in sorted(parameters)}

    def _read_production_table_from_mongo_db(self, collection_name, model_version):
        """
        Read production table from MongoDB.

        Parameters
        ----------
        collection_name: str
            Name of the collection.
        model_version: str
            Version of the model.

        Raises
        ------
        ValueError
            if query returned no results.
        """
        try:
            return DatabaseHandler.production_table_cached[
                self._cache_key(None, None, model_version, collection_name)
            ]
        except KeyError:
            pass

        query = {"model_version": model_version, "collection": collection_name}
        collection = self.get_collection(self._get_db_name(), "production_tables")
        post = collection.find_one(query)
        if not post:
            raise ValueError(f"The following query returned zero results: {query}")

        return {
            "collection": post["collection"],
            "model_version": post["model_version"],
            "parameters": post["parameters"],
            "design_model": post.get("design_model", {}),
            "entry_date": ObjectId(post["_id"]).generation_time,
        }

    def get_model_versions(self, collection_name="telescopes"):
        """
        Get list of model versions from the DB.

        Parameters
        ----------
        collection_name: str
            Name of the collection.

        Returns
        -------
        list
            List of model versions
        """
        collection = self.get_collection(self._get_db_name(), "production_tables")
        return sorted(
            [post["model_version"] for post in collection.find({"collection": collection_name})]
        )

    def get_array_elements(self, model_version, collection="telescopes"):
        """
        Get list array elements for a given model version and collection from the DB.

        Parameters
        ----------
        model_version: str
            Version of the model.
        collection: str
            Which collection to get the array elements from:
            i.e. telescopes, calibration_devices.

        Returns
        -------
        list
            Sorted list of all array elements found in collection
        """
        production_table = self._read_production_table_from_mongo_db(collection, model_version)
        return sorted([entry for entry in production_table["parameters"] if "-design" not in entry])

    def get_array_elements_of_type(self, array_element_type, model_version, collection):
        """
        Get array elements of a certain type (e.g. 'LSTN') for a DB collection.

        Does not return 'design' models.

        Parameters
        ----------
        array_element_type: str
            Type of the array element (e.g. LSTN, MSTS).
        model_version: str
            Version of the model.
        collection: str
            Which collection to get the array elements from:
            i.e. telescopes, calibration_devices.

        Returns
        -------
        list
            Sorted list of all array element names found in collection
        """
        production_table = self._read_production_table_from_mongo_db(collection, model_version)
        all_array_elements = production_table["parameters"]
        return sorted(
            [
                entry
                for entry in all_array_elements
                if entry.startswith(array_element_type) and "-design" not in entry
            ]
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
            return self.get_model_parameters(
                None,
                None,
                model_version=model_version,
                collection="configuration_corsika",
            )
        if simulation_software == "simtel":
            return (
                self.get_model_parameters(
                    site,
                    array_element_name,
                    model_version=model_version,
                    collection="configuration_sim_telarray",
                )
                if site and array_element_name
                else {}
            )
        raise ValueError(f"Unknown simulation software: {simulation_software}")

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

    def add_production_table(self, db_name, production_table):
        """
        Add a production table to the DB.

        Parameters
        ----------
        db_name: str
            the name of the DB.
        production_table: dict
            The production table to add to the DB.
        """
        db_name = self._get_db_name(db_name)
        collection = self.get_collection(db_name, "production_tables")
        self._logger.info(f"Adding production for {production_table.get('collection')} to to DB")
        collection.insert_one(production_table)
        DatabaseHandler.production_table_cached.clear()

    def add_new_parameter(
        self,
        db_name,
        par_dict,
        collection_name="telescopes",
        file_prefix=None,
    ):
        """
        Add a new parameter dictionary to the DB.

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
        par_dict = validate_data.DataValidator.validate_model_parameter(par_dict)

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

        self._reset_parameter_cache()

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

        kwargs.setdefault("content_type", "ascii/dat")
        kwargs.setdefault("filename", Path(file_name).name)

        if file_system.exists({"filename": kwargs["filename"]}):
            self._logger.warning(
                f"The file {kwargs['filename']} exists in the DB. Returning its ID"
            )
            return file_system.find_one(  # pylint: disable=protected-access
                {"filename": kwargs["filename"]}
            )._id
        self._logger.debug(f"Writing file to DB: {file_name}")
        with open(file_name, "rb") as data_file:
            return file_system.put(data_file, **kwargs)

    def _cache_key(self, site=None, array_element_name=None, model_version=None, collection=None):
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
        return "-".join(
            part for part in [model_version, collection, site, array_element_name] if part
        )

    def _read_cache(
        self, cache_dict, site=None, array_element_name=None, model_version=None, collection=None
    ):
        """
        Read parameters from cache.

        Parameters
        ----------
        cache_dict: dict
            Cache dictionary.
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
        cache_key = self._cache_key(site, array_element_name, model_version, collection)
        try:
            return cache_key, cache_dict[cache_key]
        except KeyError:
            return cache_key, None

    def _reset_parameter_cache(self):
        """Reset the cache for the parameters."""
        DatabaseHandler.model_parameters_cached.clear()

    def _get_array_element_list(self, array_element_name, site, production_table, collection):
        """
        Return list of array elements for DB queries (add design model if needed).

        Design model is added if found in the production table.

        Parameters
        ----------
        array_element_name: str
            Name of the array element.
        site: str
            Site name.
        production_table: dict
            Production table.
        collection: str
            collection of array element (e.g. telescopes, calibration_devices).

        Returns
        -------
        list
            List of array elements
        """
        if collection == "configuration_corsika":
            return ["xSTx-design"]  # placeholder to ignore 'instrument' field in query.
        if collection == "sites":
            return [f"OBS-{site}"]
        if "-design" in array_element_name:
            return [array_element_name]
        try:
            return [
                production_table["design_model"][array_element_name],
                array_element_name,
            ]
        except KeyError:
            return [
                names.get_design_model_from_name(array_element_name),
                array_element_name,
            ]
