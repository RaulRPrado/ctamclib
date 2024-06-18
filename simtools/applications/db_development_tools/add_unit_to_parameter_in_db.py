#!/usr/bin/python3

"""
    Summary
    -------
    This application is used to add a unit to parameters in the DB.

    This application should not be used by anyone but expert users and only in unusual cases.
    Therefore, no additional documentation about this applications will be given.

"""

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler


def _parse():
    config = configurator.Configurator(description="Add a unit field to a parameter in the DB.")
    return config.initialize(db_config=True, simulation_model="telescope")


def main():
    args_dict, db_config = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    # pars_to_update = ["altitude"]
    pars_to_update = ["ref_long", "ref_lat"]

    # units = ["m"]
    units = ["deg", "deg"]

    for site in ["North", "South"]:
        for par_now, unit_now in zip(pars_to_update, units):
            all_versions = db.get_all_versions(
                site=site,
                parameter=par_now,
                collection="sites",
            )
            for version_now in all_versions:
                db.update_parameter_field(
                    db_name=None,
                    site=site,
                    model_version=version_now,
                    parameter=par_now,
                    field="units",
                    new_value=unit_now,
                    collection_name="sites",
                )

                site_pars = db.get_site_parameters(site, version_now)
                assert site_pars[par_now]["units"] == unit_now


if __name__ == "__main__":
    main()
