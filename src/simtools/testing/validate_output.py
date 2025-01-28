"""Compare application output to reference output."""

import logging
from pathlib import Path

import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.testing import assertions

_logger = logging.getLogger(__name__)


def validate_all_tests(config, request, config_file_model_version):
    """
    Validate test output for all integration tests.

    Parameters
    ----------
    config: dict
        Integration test configuration dictionary.
    request: request
        Request object.
    config_file_model_version: str
        Model version from the configuration file.

    """
    if request.config.getoption("--model_version") is None:
        validate_application_output(config)
    elif config_file_model_version is not None:
        _from_command_line = request.config.getoption("--model_version")
        _from_config_file = config_file_model_version
        validate_application_output(config, _from_command_line, _from_config_file)


def validate_application_output(config, from_command_line=None, from_config_file=None):
    """
    Validate application output against expected output.

    Expected output is defined in configuration file.

    Parameters
    ----------
    config: dict
        dictionary with the configuration for the application test.
    from_command_line: str
        Model version from the command line.
    from_config_file: str
        Model version from the configuration file.

    """
    if "INTEGRATION_TESTS" not in config:
        return

    for integration_test in config["INTEGRATION_TESTS"]:
        _logger.info(f"Testing application output: {integration_test}")

        if from_command_line == from_config_file:
            if "REFERENCE_OUTPUT_FILE" in integration_test:
                _validate_reference_output_file(config, integration_test)

            if "TEST_OUTPUT_FILES" in integration_test:
                _validate_output_path_and_file(config, integration_test["TEST_OUTPUT_FILES"])

            if "OUTPUT_FILE" in integration_test:
                _validate_output_path_and_file(
                    config,
                    [{"PATH_DESCRIPTOR": "OUTPUT_PATH", "FILE": integration_test["OUTPUT_FILE"]}],
                )

            if "FILE_TYPE" in integration_test:
                assert assertions.assert_file_type(
                    integration_test["FILE_TYPE"],
                    Path(config["CONFIGURATION"]["OUTPUT_PATH"]).joinpath(
                        config["CONFIGURATION"]["OUTPUT_FILE"]
                    ),
                )
        _test_simtel_cfg_files(config, integration_test, from_command_line, from_config_file)


def _test_simtel_cfg_files(config, integration_test, from_command_line, from_config_file):
    """Test simtel cfg files."""
    if "TEST_SIMTEL_CFG_FILES" in integration_test:
        if from_command_line:
            test_simtel_cfg_file = integration_test["TEST_SIMTEL_CFG_FILES"].get(from_command_line)
        else:
            test_simtel_cfg_file = integration_test["TEST_SIMTEL_CFG_FILES"].get(from_config_file)
        if test_simtel_cfg_file:
            _validate_simtel_cfg_files(config, test_simtel_cfg_file)


def _validate_reference_output_file(config, integration_test):
    """Compare with reference output file."""
    assert compare_files(
        integration_test["REFERENCE_OUTPUT_FILE"],
        Path(config["CONFIGURATION"]["OUTPUT_PATH"]).joinpath(
            config["CONFIGURATION"]["OUTPUT_FILE"]
        ),
        integration_test.get("TOLERANCE", 1.0e-5),
        integration_test.get("TEST_COLUMNS", None),
    )


def _validate_output_path_and_file(config, integration_file_tests):
    """Check if output paths and files exist."""
    for file_test in integration_file_tests:
        try:
            output_path = config["CONFIGURATION"][file_test["PATH_DESCRIPTOR"]]
        except KeyError as exc:
            raise KeyError(
                f"Path {file_test['PATH_DESCRIPTOR']} not found in integration test configuration."
            ) from exc

        output_file_path = Path(output_path) / file_test["FILE"]
        _logger.info(f"Checking path: {output_file_path}")
        assert output_file_path.exists()

        if "EXPECTED_OUTPUT" in file_test:
            assert assertions.check_output_from_sim_telarray(
                output_file_path,
                file_test["EXPECTED_OUTPUT"],
            )


def compare_files(file1, file2, tolerance=1.0e-5, test_columns=None):
    """
    Compare two files of file type ecsv, json or yaml.

    Parameters
    ----------
    file1: str
        First file to compare
    file2: str
        Second file to compare
    tolerance: float
        Tolerance for comparing numerical values.
    test_columns: list
        List of columns to compare. If None, all columns are compared.

    Returns
    -------
    bool
        True if the files are equal.

    """
    _file1_suffix = Path(file1).suffix
    _file2_suffix = Path(file2).suffix
    if _file1_suffix != _file2_suffix:
        raise ValueError(f"File suffixes do not match: {file1} and {file2}")
    if _file1_suffix == ".ecsv":
        return compare_ecsv_files(file1, file2, tolerance, test_columns)
    if _file1_suffix in (".json", ".yaml", ".yml"):
        return compare_json_or_yaml_files(file1, file2)

    _logger.warning(f"Unknown file type for files: {file1} and {file2}")
    return False


def compare_json_or_yaml_files(file1, file2, tolerance=1.0e-2):
    """
    Compare two json or yaml files.

    Take into account float comparison for sim_telarray string-embedded floats.

    Parameters
    ----------
    file1: str
        First file to compare
    file2: str
        Second file to compare
    tolerance: float
        Tolerance for comparing numerical values.

    Returns
    -------
    bool
        True if the files are equal.

    """
    data1 = gen.collect_data_from_file(file1)
    data2 = gen.collect_data_from_file(file2)

    _logger.debug(f"Comparing json/yaml files: {file1} and {file2}")

    if data1 == data2:
        return True

    value_list_1 = data1 if isinstance(data1, list) else None
    value_list_2 = data2 if isinstance(data2, list) else None
    if "value" in data1 and isinstance(data1["value"], str):
        value_list_1 = gen.convert_string_to_list(data1.pop("value"))
    if "value" in data2 and isinstance(data2["value"], str):
        value_list_2 = gen.convert_string_to_list(data2.pop("value"))
    if value_list_1 is not None and value_list_2 is not None:
        return np.allclose(value_list_1, value_list_2, rtol=tolerance)
    return data1 == data2


def compare_ecsv_files(file1, file2, tolerance=1.0e-5, test_columns=None):
    """
    Compare two ecsv files.

    The comparison is successful if:

    - same number of rows
    - numerical values in columns are close

    The comparison can be restricted to a subset of columns with some additional
    cuts applied. This is configured through the test_columns parameter. This is
    a list of dictionaries, where each dictionary contains the following
    key-value pairs:
    - TEST_COLUMN_NAME: column name to compare.
    - CUT_COLUMN_NAME: column for filtering.
    - CUT_CONDITION: condition for filtering.

    Parameters
    ----------
    file1: str
        First file to compare
    file2: str
        Second file to compare
    tolerance: float
        Tolerance for comparing numerical values.
    test_columns: list
        List of columns to compare. If None, all columns are compared.

    """
    _logger.info(f"Comparing files: {file1} and {file2}")
    table1 = Table.read(file1, format="ascii.ecsv")
    table2 = Table.read(file2, format="ascii.ecsv")

    if test_columns is None:
        test_columns = [{"TEST_COLUMN_NAME": col} for col in table1.colnames]

    def generate_mask(table, column, condition):
        """Generate a boolean mask based on the condition (note the usage of eval)."""
        return (
            eval(f"table['{column}'] {condition}")  # pylint: disable=eval-used
            if condition
            else np.ones(len(table), dtype=bool)
        )

    for col_dict in test_columns:
        col_name = col_dict["TEST_COLUMN_NAME"]
        mask1 = generate_mask(
            table1, col_dict.get("CUT_COLUMN_NAME", ""), col_dict.get("CUT_CONDITION", "")
        )
        mask2 = generate_mask(
            table2, col_dict.get("CUT_COLUMN_NAME", ""), col_dict.get("CUT_CONDITION", "")
        )
        table1_masked, table2_masked = table1[mask1], table2[mask2]

        if len(table1_masked) != len(table2_masked):
            return False

        if np.issubdtype(table1_masked[col_name].dtype, np.floating):
            if not np.allclose(table1_masked[col_name], table2_masked[col_name], rtol=tolerance):
                _logger.warning(f"Column {col_name} outside of relative tolerance {tolerance}")
                return False

    return True


def _validate_simtel_cfg_files(config, simtel_cfg_file):
    """
    Check sim_telarray configuration files and compare with reference file.

    Note the finetuned naming of configuration files by simtools.

    """
    reference_file = Path(simtel_cfg_file)
    test_file = Path(config["CONFIGURATION"]["OUTPUT_PATH"]) / reference_file.name.replace(
        "_test", f"_{config['CONFIGURATION']['LABEL']}"
    )
    _logger.info(
        f"Comparing simtel cfg files: {reference_file} and {test_file} "
        f"for model version {config['CONFIGURATION']['MODEL_VERSION']}"
    )
    return _compare_simtel_cfg_files(reference_file, test_file)


def _compare_simtel_cfg_files(reference_file, test_file):
    """
    Compare two sim_telarray configuration files.

    Line-by-line string comparison. Requires similar sequence of
    parameters in the files. Ignore lines containing 'config_release'
    (as it contains the simtools package version).

    Parameters
    ----------
    reference_file: Path
        Reference sim_telarray configuration file.
    test_file: Path
        Test sim_telarray configuration file.

    Returns
    -------
    bool
        True if the files are equal.

    """
    with open(reference_file, encoding="utf-8") as f1, open(test_file, encoding="utf-8") as f2:
        reference_cfg = [line.rstrip() for line in f1 if line.strip()]
        test_cfg = [line.rstrip() for line in f2 if line.strip()]

    if len(reference_cfg) != len(test_cfg):
        _logger.error(
            f"Line counts differ: {reference_file} ({len(reference_cfg)} lines), "
            f"{test_file} ({len(test_cfg)} lines)."
        )
        return False

    for ref_line, test_line in zip(reference_cfg, test_cfg):
        if any(ignore in ref_line for ignore in ("config_release", "Label")):
            continue
        if ref_line != test_line:
            _logger.error(
                f"Configuration files {reference_file} and {test_file} do not match: "
                f"'{ref_line}' and '{test_line}'"
            )
            return False

    return True
