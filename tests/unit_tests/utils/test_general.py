#!/usr/bin/python3

import gzip
import logging
import os
import time
from copy import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from astropy.table import Table

import simtools.utils.general as gen
from simtools.constants import MODEL_PARAMETER_METASCHEMA

url_desy = "https://www.desy.de"
url_simtools = "https://raw.githubusercontent.com/gammasim/simtools/main/"


test_data = "Test data"


def test_collect_dict_data(io_handler, caplog) -> None:
    dict_for_yaml = {"k3": {"kk3": 4, "kk4": 3.0}, "k4": ["bla", 2]}
    test_yaml_file = io_handler.get_output_file(file_name="test_collect_dict_data.yml")
    if not Path(test_yaml_file).exists():
        with open(test_yaml_file, "w") as output:
            yaml.dump(dict_for_yaml, output, sort_keys=False)

    d2 = gen.collect_data_from_file(test_yaml_file)
    assert "k3" in d2.keys()
    assert d2["k4"] == ["bla", 2]

    _lines = gen.collect_data_from_file("tests/resources/test_file.list")
    assert len(_lines) == 2

    # astropy-type yaml file
    _file = "tests/resources/corsikaConfigTest_astropy_headers.yml"
    _dict = gen.collect_data_from_file(_file)
    assert isinstance(_dict, dict)
    assert len(_dict) > 0

    # file with several documents
    _list = gen.collect_data_from_file(MODEL_PARAMETER_METASCHEMA)
    assert isinstance(_list, list)
    assert len(_list) > 0

    # file with several documents - get first document
    _dict = gen.collect_data_from_file(MODEL_PARAMETER_METASCHEMA, 0)
    assert _dict["version"] != "0.1.0"

    with pytest.raises(gen.InvalidConfigDataError, match=r"^YAML file"):
        gen.collect_data_from_file(MODEL_PARAMETER_METASCHEMA, 999)

    # document type not supported
    assert (
        gen.collect_data_from_file(
            "tests/resources/run1_proton_za20deg_azm0deg_North_1LST_test-lst-array.corsika.zst"
        )
        is None
    )


def test_collect_dict_from_url(io_handler) -> None:
    _file = "tests/resources/num_gains.schema.yml"
    _reference_dict = gen.collect_data_from_file(_file)

    _url = url_simtools
    _url_dict = gen.collect_data_from_http(_url + _file)

    assert _reference_dict == _url_dict

    _dict = gen.collect_data_from_file(_url + _file)
    assert isinstance(_dict, dict)
    assert len(_dict) > 0

    _url = "https://raw.githubusercontent.com/gammasim/simtools/not_main/"
    with pytest.raises(FileNotFoundError):
        gen.collect_data_from_http(_url + _file)

    # yaml file with astropy header
    _url = url_simtools
    _url_dict = gen.collect_data_from_http(
        _url + "tests/resources/corsikaConfigTest_astropy_headers.yml"
    )
    assert isinstance(_url_dict, dict)
    assert len(_dict) > 0

    # simple list
    _url = url_simtools
    _url_list = gen.collect_data_from_http(_url + "tests/resources/test_file.list")
    assert isinstance(_url_list, list)
    assert len(_url_list) == 2


def test_program_is_executable(caplog) -> None:
    # (assume 'ls' exist on any system the test is running)
    assert gen.program_is_executable("/bin/ls") is not None  # The actual path should not matter
    assert gen.program_is_executable("this_program_probably_does_not_exist") is None
    with caplog.at_level("WARNING"):
        os.environ.pop("PATH", None)
    assert gen.program_is_executable("this_program_probably_does_not_exist") is None
    assert "PATH environment variable is not set." in caplog.text


def test_get_file_age(tmp_test_directory) -> None:
    # Create a temporary file and wait for 1 seconds before accessing it
    with open(tmp_test_directory / "test_file.txt", "w", encoding="utf-8") as file:
        file.write(test_data)

    time.sleep(0.2)

    try:
        age_in_minutes = gen.get_file_age(tmp_test_directory / "test_file.txt")
        # Age should be within an acceptable range (0 to 0.05 minutes or 3 seconds)
        assert 0 <= age_in_minutes <= 0.05
    except FileNotFoundError:
        pytest.fail("get_file_age raised FileNotFoundError for an existing file.")

    # Ensure that the function raises FileNotFoundError for a non-existent file
    with pytest.raises(FileNotFoundError):
        gen.get_file_age(tmp_test_directory / "nonexistent_file.txt")


def test_get_log_excerpt(tmp_test_directory) -> None:
    log_file = tmp_test_directory / "log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("This is a log file.\n")
        f.write("This is the second line of the log file.\n")

    assert gen.get_log_excerpt(log_file) == (
        "\n\nRuntime error - See below the relevant part of the log/err file.\n\n"
        f"{log_file}\n"
        "====================================================================\n\n"
        "This is a log file."
        "This is the second line of the log file.\n\n"
        "====================================================================\n"
    )


def test_file_has_text(tmp_test_directory, caplog, file_has_text) -> None:
    """Test the file_has_text function."""

    # Test with file that has text.
    file = tmp_test_directory / "test_file_has_text.txt"
    text = "test"
    with open(file, "w") as f:
        f.write(text)
    assert file_has_text(file, text)
    assert not file_has_text(file, "test2")

    # Test with empty file.
    file = tmp_test_directory / "test_file_is_empty.txt"
    with open(file, "w") as f:
        f.write("")
    assert not file_has_text(file, text)

    # Test with file that does not exist.
    file = tmp_test_directory / "test_file_does_not_exist.txt"
    assert not file_has_text(file, text)


def test_collect_kwargs() -> None:
    """
    Test the collect_kwargs function.
    """

    # Test with no kwargs.
    kwargs = {}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {}

    # Test with one kwargs.
    kwargs = {"label_a": 1}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {"a": 1}

    # Test with multiple kwargs.
    kwargs = {"label_a": 1, "label_b": 2, "label_c": 3}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {"a": 1, "b": 2, "c": 3}

    # Test with kwargs where only one starts with label_.
    kwargs = {"a": 1, "b": 2, "label_c": 3, "d": 4}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {"c": 3}

    # Test with kwargs that do not start with label_.
    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    out_kwargs = gen.collect_kwargs("label", kwargs)
    assert out_kwargs == {}


def test_set_default_kwargs() -> None:
    """
    Test the set_default_kwargs function.
    """

    in_kwargs = {"a": 1, "b": 2}
    out_kwargs = gen.set_default_kwargs(in_kwargs, c=3, d=4)
    assert out_kwargs == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_collect_final_lines(tmp_test_directory) -> None:
    """
    Test the collect_final_lines function.
    """

    # Test with no file.
    with pytest.raises(FileNotFoundError):
        gen.collect_final_lines("no_such_file.txt", 10)

    # Test with empty file.
    _file = tmp_test_directory / "empty_file.txt"
    Path(_file).touch()
    assert gen.collect_final_lines(_file, 10) == ""

    # Test with one line file.
    _file = tmp_test_directory / "one_line_file.txt"
    with open(_file, "w") as f:
        f.write("Line 1")
    assert gen.collect_final_lines(_file, 1) == "Line 1"

    # In the following tests the \n in the output are removed, but in the actual print statements,
    # where the original function is used, they are still present in the string representation.

    # Test with multiple lines file.
    _file = tmp_test_directory / "multiple_lines_file.txt"
    with open(_file, "w") as f:
        f.write("Line 1\nLine 2\nLine 3")
    assert gen.collect_final_lines(_file, 2) == "Line 2Line 3"

    # Test with file with n_lines lines.
    _file = tmp_test_directory / "n_lines_file.txt"
    with open(_file, "w") as f:
        for i in range(10):
            f.write(f"Line {i}\n")
        f.write("Line 10")
    assert gen.collect_final_lines(_file, 3) == "Line 8Line 9Line 10"

    # Test with file compressed in gzip.
    _file = tmp_test_directory / "compressed_file.txt.gz"
    with gzip.open(_file, "wb") as f:
        f.write(b"Line 1\nLine 2\nLine 3")
    assert gen.collect_final_lines(_file, 2) == "Line 2Line 3"


def test_log_level_from_user() -> None:
    """
    Test get_log_level_from_user() function.
    """
    assert gen.get_log_level_from_user("info") == logging.INFO
    assert gen.get_log_level_from_user("debug") == logging.DEBUG
    assert gen.get_log_level_from_user("warn") == logging.WARNING
    assert gen.get_log_level_from_user("warning") == logging.WARNING
    assert gen.get_log_level_from_user("error") == logging.ERROR
    assert gen.get_log_level_from_user("critical") == logging.CRITICAL

    with pytest.raises(ValueError, match=r"^'invalid' is not a logging level"):
        gen.get_log_level_from_user("invalid")
    with pytest.raises(ValueError, match=r"^'1' is not a logging level"):
        gen.get_log_level_from_user(1)
    with pytest.raises(ValueError, match=r"^'None' is not a logging level"):
        gen.get_log_level_from_user(None)
    with pytest.raises(ValueError, match=r"^'True' is not a logging level"):
        gen.get_log_level_from_user(True)


def test_copy_as_list() -> None:
    """
    Test the copy_as_list function.
    """

    # Test with a string.
    assert gen.copy_as_list("str") == ["str"]

    # Test with a list.
    assert gen.copy_as_list([1, 2, 3]) == [1, 2, 3]

    # Test with a tuple.
    assert gen.copy_as_list((1, 2, 3)) == [1, 2, 3]

    # Test with a dictionary (probably not really a useful case, but should test anyway).
    assert gen.copy_as_list({"a": 1, "b": 2}) == ["a", "b"]

    # Test with a non-iterable object.
    assert gen.copy_as_list(123) == [123]


def test_find_file_in_current_directory(tmp_test_directory) -> None:
    """
    Test finding a file in the temp test directory directory.
    """
    file_name = tmp_test_directory / "test.txt"
    with open(file_name, "w") as _file:
        _file.write(test_data)
    file_path = gen.find_file(file_name, tmp_test_directory)
    assert file_path == file_name


def test_find_file_in_non_existing_directory(tmp_test_directory) -> None:
    """
    Test finding a file in a non-existing directory.
    """
    file_name = tmp_test_directory / "test.txt"

    loc = Path("non_existing_directory")
    with pytest.raises(FileNotFoundError):
        gen.find_file(file_name, loc)


def test_find_file_recursively(tmp_test_directory) -> None:
    """
    Test finding a file recursively.
    """
    file_name = "test_1.txt"
    test_directory_sub_dir = tmp_test_directory / "test"
    Path(test_directory_sub_dir).mkdir(parents=True, exist_ok=True)
    with open(test_directory_sub_dir / file_name, "w", encoding="utf-8") as _file:
        _file.write(test_data)
    loc = tmp_test_directory
    file_path = gen.find_file(file_name, loc)
    assert file_path == Path(loc).joinpath("test").joinpath(file_name)

    # Test also the case in which we recursively find unrelated files.
    file_name = "test_2.txt"
    Path(test_directory_sub_dir / "unrelated_sub_dir").mkdir(parents=True, exist_ok=True)
    with open(
        test_directory_sub_dir / "unrelated_sub_dir" / "unrelated_file.txt", "w", encoding="utf-8"
    ) as file:
        file.write(test_data)
    loc = tmp_test_directory
    with pytest.raises(FileNotFoundError):
        gen.find_file(file_name, loc)


def test_find_file_not_found(tmp_test_directory) -> None:
    """
    Test finding a file that does not exist.
    """
    file_name = "not_existing_file.txt"
    loc = Path(tmp_test_directory)
    with pytest.raises(FileNotFoundError):
        gen.find_file(file_name, loc)


def test_is_url():
    url = url_desy
    assert gen.is_url(url) is True

    url = "sftp://www.desy.de"
    assert gen.is_url(url) is True

    url = ""
    assert gen.is_url(url) is False

    url = "https://"
    assert gen.is_url(url) is False

    url = "desy.de"
    assert gen.is_url(url) is False

    assert gen.is_url(5.0) is False


def test_collect_data_dict_from_json():
    _file = "tests/resources/reference_point_altitude.json"
    data = gen.collect_data_from_file(_file)
    assert len(data) == 6
    assert data["unit"] == "m"


def test_collect_data_from_http():
    _file = "tests/resources/num_gains.schema.yml"
    url = url_simtools

    data = gen.collect_data_from_http(url + _file)
    assert isinstance(data, dict)

    _file = "tests/resources/reference_point_altitude.json"
    data = gen.collect_data_from_http(url + _file)
    assert isinstance(data, dict)

    _file = "tests/resources/simtel_output_files.txt"
    with pytest.raises(TypeError):
        gen.collect_data_from_http(url + _file)

    url = "https://raw.githubusercontent.com/gammasim/simtools/not_right/"
    with pytest.raises(FileNotFoundError):
        gen.collect_data_from_http(url + _file)


def test_join_url_or_path():
    assert gen.join_url_or_path(url_desy, "test") == "https://www.desy.de/test"
    assert gen.join_url_or_path(url_desy, "test", "test") == "https://www.desy.de/test/test"
    assert gen.join_url_or_path("/Volume/fs01", "CTA") == Path("/Volume/fs01").joinpath("CTA")


def test_change_dict_keys_case(caplog) -> None:
    # note that entries in DATA_COLUMNS:ATTRIBUTE should not be changed (not keys)
    _upper_dict = {
        "REFERENCE": {"VERSION": "0.1.0"},
        "ACTIVITY": {"NAME": "submit", "ID": "84890304", "DESCRIPTION": "Set data"},
        "DATA_COLUMNS": {"ATTRIBUTE": ["remove_duplicates", "SORT"]},
        "DICT_IN_LIST": {
            "KEY_OF_FIRST_DICT": ["FIRST_ITEM", {"KEY_OF_NESTED_DICT": "VALUE_OF_SECOND_DICT"}]
        },
    }
    _lower_dict = {
        "reference": {"version": "0.1.0"},
        "activity": {"name": "submit", "id": "84890304", "description": "Set data"},
        "data_columns": {"attribute": ["remove_duplicates", "SORT"]},
        "dict_in_list": {
            "key_of_first_dict": ["FIRST_ITEM", {"key_of_nested_dict": "VALUE_OF_SECOND_DICT"}]
        },
    }
    _no_change_dict_upper = gen.change_dict_keys_case(copy(_upper_dict), False)
    assert _no_change_dict_upper == _upper_dict

    _no_change_dict_lower = gen.change_dict_keys_case(copy(_lower_dict), True)
    assert _no_change_dict_lower == _lower_dict

    _changed_to_lower = gen.change_dict_keys_case(copy(_upper_dict), True)
    assert _changed_to_lower == _lower_dict

    _changed_to_upper = gen.change_dict_keys_case(copy(_lower_dict), False)
    assert _changed_to_upper == _upper_dict
    with caplog.at_level("ERROR"):
        with pytest.raises(AttributeError):
            gen.change_dict_keys_case([2], False)
    assert "Input is not a proper dictionary" in caplog.text


def test_sort_arrays() -> None:
    """
    Test the sort_arrays function.
    """

    # Test with no arguments.
    args = []
    new_args = gen.sort_arrays(*args)
    assert not new_args

    # Test with one argument.
    args = [list(range(10))]
    new_args = gen.sort_arrays(*args)
    assert new_args == [list(range(10))]

    # Test with multiple arguments.
    args = [list(range(10)), list(range(10, 20))]
    new_args = gen.sort_arrays(*args)
    assert new_args == [list(range(10)), list(range(10, 20))]

    # Test with arguments of different lengths.
    args = [list(range(10)), list(range(5))]
    new_args = gen.sort_arrays(*args)
    assert new_args == [list(range(10)), list(range(5))]

    # Test with arguments that are not arrays.
    args = [1, 2, 3]
    with pytest.raises(TypeError):
        gen.sort_arrays(*args)

    # Test with the input array not in the right order.
    args = [list(reversed(range(10)))]
    new_args = gen.sort_arrays(*args)
    assert new_args == [list(range(10))]


@pytest.mark.parametrize(
    ("input_data", "expected_output"),
    [
        (
            {
                "key1": "This is a string\n with a newline",
                "key2": ["List item 1\n", "List item 2\n"],
                "key3": {"nested_key": "Nested string\n with a newline"},
                "key4": [{"nested_dict": "string2\n"}, {"nested_dict2": "string3\n"}],
            },
            {
                "key1": "This is a string with a newline",
                "key2": ["List item 1", "List item 2"],
                "key3": {"nested_key": "Nested string with a newline"},
                "key4": [{"nested_dict": "string2"}, {"nested_dict2": "string3"}],
            },
        ),
    ],
)
def test_remove_substring_recursively_from_dict(input_data, expected_output, caplog):
    result = gen.remove_substring_recursively_from_dict(input_data, "\n")
    assert result == expected_output

    # no error should be raised for None input, but a debug message should be printed
    gen._logger.setLevel(logging.DEBUG)
    gen.remove_substring_recursively_from_dict([2])
    assert any(
        record.levelname == "DEBUG" and "Input is not a dictionary: [2]" in record.message
        for record in caplog.records
    )


@patch("builtins.input", side_effect=["Y", "y"])
def test_user_confirm_yes(mock_input):
    assert gen.user_confirm()


@patch("builtins.input", side_effect=["N", "n", EOFError, "not_Y_or_N"])
def test_user_confirm_no(mock_input):
    assert not gen.user_confirm()


def test_validate_data_type():

    test_cases = [
        # Test exact data type match
        ("int", 5, None, False, True),
        ("int", 5.5, None, False, False),
        ("float", 3.14, None, False, True),
        ("str", "hello", None, False, True),
        ("bool", True, None, False, True),
        ("bool", 1, None, False, False),
        ("int", None, int, False, True),
        ("float", None, float, False, True),
        ("str", None, str, False, True),
        ("bool", None, bool, False, True),
        ("bool", None, bool, False, True),
        # Test allow_subtypes=True
        ("float", 5, None, True, True),
        ("float", [1, 2, 3], None, True, True),
        ("int", [1, 2, 3], None, True, True),
        ("int", np.array([1, 2, 3]), None, True, True),
        ("float", np.array([1.0, 2.0, 3.0]), None, True, True),
        ("file", "hello", None, True, True),
        ("string", "hello", None, True, True),
        ("file", None, "object", True, True),  # 'file' type with None value
        ("boolean", True, None, True, True),
        ("int", None, np.uint8, True, True),  # Subtype of 'int'
        ("float", None, int, True, True),  # 'int' can be converted to 'float'
    ]

    for reference_dtype, value, dtype, allow_subtypes, expected_result in test_cases:
        gen._logger.debug(f"{reference_dtype} {value} {dtype} {allow_subtypes} {expected_result}")
        assert (
            gen.validate_data_type(
                reference_dtype=reference_dtype,
                value=value,
                dtype=dtype,
                allow_subtypes=allow_subtypes,
            )
            is expected_result
        )

    with pytest.raises(ValueError, match=r"^Either value or dtype must be given"):
        gen.validate_data_type("int", None, None, False)


def test_convert_list_to_string():

    assert gen.convert_list_to_string(None) is None
    assert gen.convert_list_to_string("a") == "a"
    assert gen.convert_list_to_string(5) == 5
    assert gen.convert_list_to_string([1, 2, 3]) == "1 2 3"
    assert gen.convert_list_to_string(np.array([1, 2, 3])) == "1 2 3"
    assert gen.convert_list_to_string(np.array([1, 2, 3]), comma_separated=True) == "1, 2, 3"
    assert (
        gen.convert_list_to_string(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], comma_separated=False, shorten_list=True
        )
        == "all: 1"
    )
    assert (
        gen.convert_list_to_string(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], comma_separated=False, shorten_list=False
        )
        == "1 1 1 1 1 1 1 1 1 1 1"
    )
    assert (
        gen.convert_list_to_string(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            comma_separated=False,
            shorten_list=False,
            collapse_list=True,
        )
        == "1"
    )
    assert gen.convert_list_to_string([1, 2, 3], collapse_list=True) == "1 2 3"
    assert gen.convert_list_to_string([1, 2, 3], shorten_list=True) == "1 2 3"


def test_convert_string_to_list():

    t_1 = gen.convert_string_to_list("1 2 3 4")
    assert len(t_1) == 4
    assert pytest.approx(t_1[1]) == 2.0
    assert isinstance(t_1[1], float)

    t_int = gen.convert_string_to_list("1 2 3 4", False)
    assert len(t_int) == 4
    assert t_int[1] == 2
    assert isinstance(t_int[1], int)

    t_2 = gen.convert_string_to_list("0.1 0.2 0.3 0.4")
    assert len(t_2) == 4
    assert pytest.approx(t_2[1]) == 0.2

    t_3 = gen.convert_string_to_list("0.1")
    assert pytest.approx(t_3[0]) == 0.1

    assert gen.convert_string_to_list("bla_bla") == "bla_bla"
    assert gen.convert_string_to_list("bla bla") == ["bla", "bla"]
    assert gen.convert_string_to_list("bla,bla") == ["bla", "bla"]
    # import for list of dimensionless entries in database
    assert gen.convert_string_to_list(",") == ["", ""]
    assert gen.convert_string_to_list(" , , ") == ["", "", ""]


def test_read_file_encoded_in_utf_or_latin(tmp_test_directory, caplog) -> None:
    """
    Test the read_file_encoded_in_utf_or_latin function.
    """

    # Test with a UTF-8 encoded file.
    utf8_file = tmp_test_directory / "utf8_file.txt"
    utf8_content = "This is a UTF-8 encoded file.\n"
    with open(utf8_file, "w", encoding="utf-8") as file:
        file.write(utf8_content)
    lines = gen.read_file_encoded_in_utf_or_latin(utf8_file)
    assert lines == [utf8_content]

    # Test with a Latin-1 encoded file.
    latin1_file = tmp_test_directory / "latin1_file.txt"
    latin1_content = "This is a Latin-1 encoded file with latin character ñ.\n"
    with open(latin1_file, "w", encoding="latin-1") as file:
        file.write(latin1_content)
    with caplog.at_level(logging.DEBUG):
        lines = gen.read_file_encoded_in_utf_or_latin(latin1_file)
        assert lines == [latin1_content]
    assert "Unable to decode file using UTF-8. Trying Latin-1." in caplog.text

    # I could not find a way to create a file that cannot be decoded with Latin-1
    # and raises a UnicodeDecodeError. I left the raise statement in the function
    # in case we ever encounter such a file, but I cannot test it here.

    # Test with a non-existent file.
    non_existent_file = tmp_test_directory / "non_existent_file.txt"
    with pytest.raises(FileNotFoundError):
        gen.read_file_encoded_in_utf_or_latin(non_existent_file)


def test_get_structure_array_from_table():
    table = Table(
        {
            "col1": [1, 2, 3],
            "col2": [4.0, 5.0, 6.0],
            "col3": ["a", "b", "c"],
        }
    )

    # Test with all columns
    column_names = ["col1", "col2", "col3"]
    structured_array = gen.get_structure_array_from_table(table, column_names)
    assert structured_array.dtype.names == ("col1", "col2", "col3")
    assert structured_array["col1"].tolist() == [1, 2, 3]
    assert structured_array["col2"].tolist() == [4.0, 5.0, 6.0]
    assert structured_array["col3"].tolist() == ["a", "b", "c"]

    # Test with a subset of columns
    column_names = ["col1", "col3"]
    structured_array = gen.get_structure_array_from_table(table, column_names)
    assert structured_array.dtype.names == ("col1", "col3")
    assert structured_array["col1"].tolist() == [1, 2, 3]
    assert structured_array["col3"].tolist() == ["a", "b", "c"]

    # Test with a single column
    column_names = ["col2"]
    structured_array = gen.get_structure_array_from_table(table, column_names)
    assert structured_array.dtype.names == ("col2",)
    assert structured_array["col2"].tolist() == [4.0, 5.0, 6.0]

    # Test with an empty list of columns
    column_names = []
    assert gen.get_structure_array_from_table(table, column_names).size == 0

    # Test with a non-existent column
    column_names = ["col1", "non_existent_col"]
    with pytest.raises(KeyError):
        gen.get_structure_array_from_table(table, column_names)


def test_convert_keys_in_dict_to_lowercase():
    """
    Test the convert_keys_in_dict_to_lowercase function.
    """

    # Test with a simple dictionary.
    input_data = {"Key1": "value1", "Key2": "value2"}
    expected_output = {"key1": "value1", "key2": "value2"}
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a nested dictionary.
    input_data = {"Key1": {"NestedKey1": "value1"}, "Key2": "value2"}
    expected_output = {"key1": {"nestedkey1": "value1"}, "key2": "value2"}
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a dictionary containing a list.
    input_data = {"Key1": ["Value1", {"NestedKey1": "value1"}], "Key2": "value2"}
    expected_output = {"key1": ["Value1", {"nestedkey1": "value1"}], "key2": "value2"}
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a list of dictionaries.
    input_data = [{"Key1": "value1"}, {"Key2": "value2"}]
    expected_output = [{"key1": "value1"}, {"key2": "value2"}]
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a non-dictionary input.
    input_data = "String"
    expected_output = "String"
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with an empty dictionary.
    input_data = {}
    expected_output = {}
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output

    # Test with a dictionary containing mixed types.
    input_data = {"Key1": 123, "Key2": [1, 2, 3], "Key3": {"NestedKey1": "value1"}}
    expected_output = {"key1": 123, "key2": [1, 2, 3], "key3": {"nestedkey1": "value1"}}
    assert gen.convert_keys_in_dict_to_lowercase(input_data) == expected_output


def test_clear_default_sim_telarray_cfg_directories():
    """
    Test the clear_default_sim_telarray_cfg_directories function.
    """

    # Test with a simple command.
    command = "run_simulation"
    expected_output = "SIM_TELARRAY_CONFIG_PATH='' run_simulation"
    assert gen.clear_default_sim_telarray_cfg_directories(command) == expected_output

    # Test with a command containing spaces.
    command = "run_simulation --config config_file"
    expected_output = "SIM_TELARRAY_CONFIG_PATH='' run_simulation --config config_file"
    assert gen.clear_default_sim_telarray_cfg_directories(command) == expected_output

    # Test with an empty command.
    command = ""
    expected_output = "SIM_TELARRAY_CONFIG_PATH='' "
    assert gen.clear_default_sim_telarray_cfg_directories(command) == expected_output

    # Test with a command containing special characters.
    command = "run_simulation && echo 'done'"
    expected_output = "SIM_TELARRAY_CONFIG_PATH='' run_simulation && echo 'done'"
    assert gen.clear_default_sim_telarray_cfg_directories(command) == expected_output
