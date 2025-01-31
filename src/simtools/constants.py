"""Project wide constants."""

from importlib.resources import files

# Schema path
SCHEMA_PATH = files("simtools") / "schemas"
# Path to metadata jsonschema
METADATA_JSON_SCHEMA = SCHEMA_PATH / "metadata.metaschema.yml"
# Path to model parameter metaschema
MODEL_PARAMETER_METASCHEMA = SCHEMA_PATH / "model_parameter.metaschema.yml"
# Path to model parameter description metaschema
MODEL_PARAMETER_DESCRIPTION_METASCHEMA = (
    SCHEMA_PATH / "model_parameter_and_data_schema.metaschema.yml"
)
# Path to model parameter schema files
MODEL_PARAMETER_SCHEMA_PATH = SCHEMA_PATH / "model_parameters"
