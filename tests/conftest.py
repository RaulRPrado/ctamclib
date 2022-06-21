import os
import logging
from pathlib import Path
import pytest
import yaml

import simtools.config as cfg


logger = logging.getLogger()


def write_configuration_test_file(config_file, config_dict):
    """
    Write a simtools configuration file

    Do not overwrite existing files
    """

    if not Path(config_file).exists():
        with open(config_file, "w") as output:
            yaml.safe_dump(
                config_dict,
                output,
                sort_keys=False)


@pytest.fixture
def tmp_test_directory(tmpdir_factory):
    tmp_test_dir = tmpdir_factory.mktemp("test-data")
    tmp_sub_dirs = ['/resources/', '/output/', '/simtel/']
    for sub_dir in tmp_sub_dirs:
        tmp_sub_dir = tmp_test_dir + sub_dir
        tmp_sub_dir.mkdir()

    return tmp_test_dir


@pytest.fixture
def configuration_parameters(tmp_test_directory):

    return {
        'modelFilesLocations':  [
            str(tmp_test_directory) + '/resources/',
            './tests/resources/'
        ],
        'dataLocation': './data/',
        'outputLocation': str(tmp_test_directory) + '/output/',
        'simtelPath': str(tmp_test_directory) + '/simtel/',
        'useMongoDB': False,
        'mongoDBConfigFile': None,
        'extraCommands': [''],
    }


@pytest.fixture
def db_connection(tmp_test_directory):
    # prefer local dbDetails.yml file for testing
    try:
        dbDetailsFile = "dbDetails.yml"
        with open(dbDetailsFile, "r") as stream:
            yaml.safe_load(stream)
        return dbDetailsFile
    # try if DB details are defined in environment
    # (e.g., as secrets in github actions)
    except FileNotFoundError:
        parsToDbDetails = {}
        environVariblesToCheck = {
            "mongodbServer": "DB_API_NAME",
            "userDB": "DB_API_USER",
            "passDB": "DB_API_PW",
            "dbPort": "DB_API_PORT",
        }
        found_env = True
        for par, env in environVariblesToCheck.items():
            if env in os.environ:
                parsToDbDetails[par] = os.environ[env]
            else:
                found_env = False
        if found_env:
            dbDetailsFileName = str(tmp_test_directory) + "dbDetails.yml"
            cfg.createDummyDbDetails(filename=dbDetailsFileName, **parsToDbDetails)
            return dbDetailsFileName

        return ""


@pytest.fixture
def set_db(db_connection, tmp_test_directory, configuration_parameters):
    """
    Configuration file for using simtools
    - with database
    - without sim_telarray

    (some code duplication with cfg_setup, set_simtelarray, set_simtools, set_db)
    """

    if len(str(db_connection)) == 0:
        pytest.skip("Test requires database (DB) connection")

    config_file = str(tmp_test_directory) + '/config-db-test.yml'
    config_dict = dict(configuration_parameters)
    config_dict['useMongoDB'] = True
    config_dict['mongoDBConfigFile'] = str(db_connection)

    write_configuration_test_file(config_file, config_dict)
    cfg.setConfigFileName(config_file)


@pytest.fixture
def simtelpath():
    simtelpath = Path(os.path.expandvars('$SIMTELPATH'))
    if simtelpath.exists():
        return simtelpath

    return ""


@pytest.fixture
def set_simtelarray(simtelpath, tmp_test_directory, configuration_parameters):
    """
    Configuration file for using simtools
    - without database
    - with sim_telarray

    (some code duplication with cfg_setup, set_simtelarray, set_simtools, set_db)
    """

    if len(str(simtelpath)) == 0:
        pytest.skip('sim_telarray not found in {}'.format(simtelpath))

    config_file = str(tmp_test_directory) + '/config-simtelarray-test.yml'
    config_dict = dict(configuration_parameters)
    config_dict['simtelPath'] = str(simtelpath)
    write_configuration_test_file(config_file, config_dict)
    cfg.setConfigFileName(config_file)


@pytest.fixture
def set_simtools(db_connection, simtelpath, tmp_test_directory, configuration_parameters):
    """
    Configuration file for using simtools
    - with database
    - with sim_telarray

    (some code duplication with cfg_setup, set_simtelarray, set_simtools, set_db)
    """

    if len(str(simtelpath)) == 0:
        pytest.skip('sim_telarray not found in {}'.format(simtelpath))
    if len(str(db_connection)) == 0:
        pytest.skip("Test requires database (DB) connection")

    config_file = str(tmp_test_directory) + '/config-simtools-test.yml'
    config_dict = dict(configuration_parameters)
    config_dict['simtelPath'] = str(simtelpath)
    config_dict['useMongoDB'] = True
    config_dict['mongoDBConfigFile'] = str(db_connection)

    write_configuration_test_file(config_file, config_dict)
    cfg.setConfigFileName(config_file)


@pytest.fixture
def cfg_setup(tmp_test_directory, configuration_parameters):
    """
    Configuration file for using simtools
    - without database
    - without sim_telarray

    (some code duplication with cfg_setup, set_simtelarray, set_simtools, set_db)
    """

    config_file = str(tmp_test_directory) + '/config-test.yml'
    write_configuration_test_file(config_file, dict(configuration_parameters))
    cfg.setConfigFileName(config_file)
