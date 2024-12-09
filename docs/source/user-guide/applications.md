(applications)=

# simtools Applications

Applications are python scripts built on the {ref}`Library` that execute a well defined task.
Application scripts can be found in `simtools/applications`.
They are the the building blocks of [Simulation System Workflows](https://github.com/gammasim/workflows).

Important: depending on the installation type, applications are named differently:

- developers (see [installation for developers](../developer-guide/getting_started.md#devinstallationfordevelopers)) call applications as described throughout this documentation: `python applications/<application name> ....`
- users (see {ref}`InstallationForUsers`) call applications directly as command-line tool. Applications names `simtools-<application name` (with all `_` replaced by `-`)

Each application is configured as described in {ref}`Configuration`.
The available arguments can be access by calling the `python applications/<application name> --help`.

Some applications require one or multiple filenames as input from the command-line options. The system will
first search on main simtools directory for these files, and in case it is not found, it will
search into the directories given by the config parameter *model_path*.

Output files of applications are written to `$output_path/$label`, where
*output_path* is a config parameter and *label* is the name of the application. The plots
produced directly by the application are stored in the sub-directory *application-plots*.
High-level data produced intermediately (e.g PSF tables) can be found in the sub-directories relative to
the specific type of application (e.g *ray-tracing* for optics related applications,
*camera-efficiency* for camera efficiency applications etc). All files related to the simulation model (e.g,
sim_telarray config files) are stored in the sub-directory *model*.

Applications found in the *simtools/application/db_development_tools* directory are not intended for
end users, but for developers working on the database schema.

## List of applications

```{toctree}
:glob: true
:maxdepth: 1

calculate_trigger_rate <applications/calculate_trigger_rate>
convert_all_model_parameters_from_simtel <applications/convert_all_model_parameters_from_simtel>
convert_geo_coordinates_of_array_elements <applications/convert_geo_coordinates_of_array_elements>
convert_model_parameter_from_simtel <applications/convert_model_parameter_from_simtel>
db_add_file_to_db <applications/db_add_file_to_db>
db_add_model_parameters_from_repository_to_db <applications/db_add_model_parameters_from_repository_to_db>
db_add_value_from_json_to_db <applications/db_add_value_from_json_to_db>
db_get_array_layouts_from_db <applications/db_get_array_layouts_from_db>
db_get_file_from_db <applications/db_get_file_from_db>
db_get_parameter_from_db <applications/db_get_parameter_from_db>
db_inspect_databases <applications/db_inspect_databases>
derive_mirror_rnda <applications/derive_mirror_rnda>
derive_psf_parameters <applications/derive_psf_parameters>
generate_array_config <applications/generate_array_config>
generate_corsika_histograms <applications/generate_corsika_histograms>
generate_default_metadata <applications/generate_default_metadata>
generate_simtel_array_histograms <applications/generate_simtel_array_histograms>
generate_regular_arrays <applications/generate_regular_arrays>
plot_array_layout <applications/plot_array_layout>
production_generate_grid <applications/production_generate_grid>
production_scale_events <applications/production_scale_events>
production_generate_simulation_config <applications/production_generate_simulation_config>
production_calculate_resource_estimates <applications/production_calculate_resource_estimates>
simulate_light_emission <applications/simulate_light_emission>
simulate_prod <applications/simulate_prod>
submit_data_from_external <applications/submit_data_from_external>
submit_model_parameter_from_external <applications/submit_model_parameter_from_external>
validate_camera_efficiency <applications/validate_camera_efficiency>
validate_camera_fov <applications/validate_camera_fov>
validate_cumulative_psf <applications/validate_cumulative_psf>
validate_file_using_schema <applications/validate_file_using_schema>
validate_optics <applications/validate_optics>
```
