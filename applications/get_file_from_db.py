#!/usr/bin/python3

"""
    Summary
    -------
    Get a file from the DB.

    The name of the file is required.
    This application complements the ones for getting parameters, adding entries and files \
    to the DB.

    Command line arguments
    ----------------------
    file_name (str or list of str, required)
        Name of the file to get including its full directory. A list of files is also allowed.
        i.e., python applications/get_file_from_db.py -file_name mirror_CTA-N-LST1_v2019-03-31.dat.
    output_path (str)
        Name of the local output directory where to save the files.
        Default it $CWD.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    getting a file from the DB.

    .. code-block:: console

        python applications/get_file_from_db.py ---file_name mirror_CTA-N-LST1_v2019-03-31.dat
"""

import logging

import simtools.configuration.configurator as configurator
import simtools.util.general as gen
from simtools import db_handler


def main():

    config = configurator.Configurator(
        label="Get file(s) from the DB.",
        description="python applications/get_file_from_db.py "
        " --file_name mirror_CTA-S-LST_v2020-04-07.dat",
    )

    config.parser.add_argument(
        "--file_name",
        help=("The name of the file name to be downloaded."),
        type=str,
    )
    args_dict, db_config = config.initialize(db_config=True)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
    available_dbs = [
        db.DB_TABULATED_DATA,
        db.DB_CTA_SIMULATION_MODEL,
        db.DB_CTA_SIMULATION_MODEL_DESCRIPTIONS,
        db.DB_REFERENCE_DATA,
        db.DB_DERIVED_VALUES,
        "sandbox",
        "test-data",
    ]
    file_id = None
    if args_dict["output_path"].exists():
        for db_name in available_dbs:
            try:
                file_id = db.export_file_db(
                    db_name, args_dict["output_path"], args_dict["file_name"]
                )
                logger.info(
                    "Got file {} from DB {} and saved into {}".format(
                        args_dict["file_name"], db_name, args_dict["output_path"]
                    )
                )
                break
            except FileNotFoundError:
                continue

        if file_id is None:
            logger.error(
                "The file {} was not found in any of the available DBs.".format(
                    args_dict["file_name"]
                )
            )
            raise FileNotFoundError
    else:
        logger.error("Aborted, directory {} does not exist".format(args_dict["output_path"]))


if __name__ == "__main__":
    main()
