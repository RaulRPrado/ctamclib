#!/usr/bin/python3

import logging
from pathlib import Path

import pytest

import simtools.io_operations.io_handler as io_handler_module

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def test_file():
    return "test-file.txt"


def test_get_output_directory(args_dict, io_handler):
    # default adding label
    assert io_handler.get_output_directory(label="test-io-handler") == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/"
    )

    # label and testing
    assert io_handler.get_output_directory(label="test-io-handler", dir_type="test") == Path(
        f"{args_dict['output_path']}/output/test-output/test-io-handler/"
    )

    # label and subdirectory (no testing)
    assert io_handler.get_output_directory(label="test-io-handler", sub_dir="model") == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/model"
    )

    # label and subdirectory (no testing); simtools-result should have no effect
    assert io_handler.get_output_directory(
        label="test-io-handler", sub_dir="model", dir_type="simtools-result"
    ) == Path(f"{args_dict['output_path']}/output/simtools-output/test-io-handler/model")

    # label and subdirectory (testing)
    assert io_handler.get_output_directory(
        label="test-io-handler", sub_dir="model", dir_type="test"
    ) == Path(f"{args_dict['output_path']}/output/test-output/test-io-handler/model")

    with pytest.raises(TypeError):
        io_handler.get_output_directory(label="test-io-handler", sub_dir="model", dir_type=None)


def test_get_output_directory_plain_output_path(args_dict, io_handler):
    # all following tests: plain_path tests
    io_handler.use_plain_output_path = True

    # plain path (label has no effect), no subdirectories (no testing)
    assert io_handler.get_output_directory(label="test-io-handler") == Path(
        f"{args_dict['output_path']}/output"
    )

    # plain path (label has no effect), no subdirectories (testing is ignored)
    assert io_handler.get_output_directory(label="test-io-handler", dir_type="test") == Path(
        f"{args_dict['output_path']}/output"
    )

    # plain path, label has no effect, with sub directory as dir_type != 'simtools-result'
    # (no testing)
    assert io_handler.get_output_directory(label="test-io-handler", sub_dir="model") == Path(
        f"{args_dict['output_path']}/output/model"
    )

    # plain path, label has no effect, with sub directory as dir_type != 'simtools-result'
    # (testing)
    assert io_handler.get_output_directory(
        label="test-io-handler", sub_dir="model", dir_type="test"
    ) == Path(f"{args_dict['output_path']}/output/model")

    # plain path, label has no effect, without sub directory as dir_type == 'simtools-result'
    # (no testing)
    assert io_handler.get_output_directory(
        label="test-io-handler", sub_dir="model", dir_type="simtools-result"
    ) == Path(f"{args_dict['output_path']}/output/")


def test_get_output_file(args_dict, io_handler, test_file):
    assert io_handler.get_output_file(file_name=test_file, label="test-io-handler") == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/{test_file}"
    )

    assert io_handler.get_output_file(
        file_name=test_file,
        label="test-io-handler",
        dir_type="test",
    ) == Path(f"{args_dict['output_path']}/output/test-output/test-io-handler/{test_file}")

    assert io_handler.get_output_file(
        file_name=test_file,
        label="test-io-handler",
        sub_dir="model",
    ) == Path(
        f"{args_dict['output_path']}/output/simtools-output/test-io-handler/model/{test_file}"
    )

    assert io_handler.get_output_file(
        file_name=test_file,
        label="test-io-handler",
        sub_dir="model",
        dir_type="test",
    ) == Path(f"{args_dict['output_path']}/output/test-output/test-io-handler/model/{test_file}")


def test_get_data_file(args_dict, io_handler, test_file):
    assert (
        io_handler.get_input_data_file(
            parent_dir="test-io-handler",
            file_name=test_file,
        )
        == Path(f"{args_dict['data_path']}/test-io-handler/{test_file}").absolute()
    )

    assert (
        io_handler.get_input_data_file(file_name=test_file, test=True)
        == Path(f"tests/resources/{test_file}").absolute()
    )

    io_handler.data_path = None
    with pytest.raises(io_handler_module.IncompleteIOHandlerInitError):
        io_handler.get_input_data_file(file_name=test_file)
