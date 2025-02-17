#!/usr/bin/python3

import argparse
import logging

import astropy.units as u
import pytest

import simtools.configuration.commandline_parser as parser

logger = logging.getLogger()
SIMULATION_MODEL_STRING = "simulation model"


def test_site():
    assert parser.CommandLineParser.site("North") == "North"
    assert parser.CommandLineParser.site("South") == "South"

    with pytest.raises(ValueError, match=r"Invalid name East"):
        parser.CommandLineParser.site("East")


def test_telescope():
    assert parser.CommandLineParser.telescope("LSTN-01") == "LSTN-01"
    assert parser.CommandLineParser.telescope("MSTx-NectarCam") == "MSTx-NectarCam"

    with pytest.raises(ValueError, match=r"Invalid name Whipple"):
        assert parser.CommandLineParser.telescope("Whipple")

    with pytest.raises(ValueError, match=r"Invalid name LST"):
        assert parser.CommandLineParser.telescope("LST")


def test_efficiency_interval():
    assert parser.CommandLineParser.efficiency_interval(0.5) == pytest.approx(0.5)
    assert parser.CommandLineParser.efficiency_interval(0.0) == pytest.approx(0.0)
    assert parser.CommandLineParser.efficiency_interval(1.0) == pytest.approx(1.0)

    with pytest.raises(
        argparse.ArgumentTypeError, match=r"1.5 outside of allowed \[0,1\] interval"
    ):
        parser.CommandLineParser.efficiency_interval(1.5)
    with pytest.raises(
        argparse.ArgumentTypeError, match=r"-8.5 outside of allowed \[0,1\] interval"
    ):
        parser.CommandLineParser.efficiency_interval(-8.5)


def test_zenith_angle(caplog):
    assert parser.CommandLineParser.zenith_angle(0).value == pytest.approx(0.0)
    assert parser.CommandLineParser.zenith_angle(45).value == pytest.approx(45.0)
    assert parser.CommandLineParser.zenith_angle(90).value == pytest.approx(90.0)
    assert isinstance(parser.CommandLineParser.zenith_angle(0), u.Quantity)
    assert parser.CommandLineParser.zenith_angle("0 deg").value == pytest.approx(0.0)
    assert parser.CommandLineParser.zenith_angle("45 deg").value == pytest.approx(45.0)
    assert parser.CommandLineParser.zenith_angle("90 deg").value == pytest.approx(90.0)

    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided zenith angle, -1.0, is outside of the allowed \[0, 180\] interval",
    ):
        parser.CommandLineParser.zenith_angle(-1)
    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided zenith angle, 190.0, is outside of the allowed \[0, 180\] interval",
    ):
        parser.CommandLineParser.zenith_angle(190)

    with caplog.at_level("WARNING"):
        with pytest.raises(TypeError):
            parser.CommandLineParser.zenith_angle("North")
    assert "The zenith angle provided is not a valid numeric" in caplog.text


def test_parse_quantity_pair(caplog):
    for test_string in ["100 GeV 5 TeV", "100GeV 5TeV", "100GeV 5 TeV"]:
        e_pair = parser.CommandLineParser.parse_quantity_pair(test_string)
        assert pytest.approx(e_pair[0].value) == 100.0
        assert e_pair[0].unit == u.GeV
        assert pytest.approx(e_pair[1].value) == 5.0
        assert e_pair[1].unit == u.TeV

    with pytest.raises(ValueError, match=r"Input string does not contain exactly two quantities."):
        parser.CommandLineParser.parse_quantity_pair("100 GeV 5 TeV 20 PeV")

    with pytest.raises(ValueError, match=r"^'abc' did not parse as unit:"):
        parser.CommandLineParser.parse_quantity_pair("100 GeV 5 abc")

    with pytest.raises(ValueError, match=r"Input string does not contain exactly two quantities."):
        parser.CommandLineParser.parse_quantity_pair("a GeV 5 TeV")


def test_parse_integer_and_quantity(caplog):
    for test_string in ["5 1500 m", "5 1500m", "5 1500.0 m", "(5, <Quantity 1500 m>)"]:
        c_pair = parser.CommandLineParser.parse_integer_and_quantity(test_string)
        assert c_pair[0] == 5
        assert pytest.approx(c_pair[1].value) == 1500.0
        assert c_pair[1].unit == u.m

    with pytest.raises(ValueError, match=r"^'abc' did not parse as unit:"):
        parser.CommandLineParser.parse_integer_and_quantity("5 5 abc")
    with pytest.raises(
        ValueError, match=r"Input string does not contain an integer and a astropy quantity."
    ):
        parser.CommandLineParser.parse_integer_and_quantity("0 m 5 m")


def test_azimuth_angle(caplog):
    assert parser.CommandLineParser.azimuth_angle(0).value == pytest.approx(0.0)
    assert parser.CommandLineParser.azimuth_angle(45).value == pytest.approx(45.0)
    assert parser.CommandLineParser.azimuth_angle(90).value == pytest.approx(90.0)
    assert isinstance(parser.CommandLineParser.azimuth_angle(0), u.Quantity)
    assert parser.CommandLineParser.azimuth_angle("0 deg").value == pytest.approx(0.0)
    assert parser.CommandLineParser.azimuth_angle("45 deg").value == pytest.approx(45.0)
    assert parser.CommandLineParser.azimuth_angle("90 deg").value == pytest.approx(90.0)

    assert parser.CommandLineParser.azimuth_angle("North").value == pytest.approx(0.0)
    assert parser.CommandLineParser.azimuth_angle("South").value == pytest.approx(180.0)
    assert parser.CommandLineParser.azimuth_angle("East").value == pytest.approx(90.0)
    assert parser.CommandLineParser.azimuth_angle("West").value == pytest.approx(270.0)

    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided azimuth angle, -1.0, is outside of the allowed \[0, 360\] interval",
    ):
        parser.CommandLineParser.azimuth_angle(-1)
    with pytest.raises(
        argparse.ArgumentTypeError,
        match=r"The provided azimuth angle, 370.0, is outside of the allowed \[0, 360\] interval",
    ):
        parser.CommandLineParser.azimuth_angle(370)
    caplog.clear()
    with pytest.raises(
        argparse.ArgumentTypeError, match=r"^The azimuth angle given as string can only be one of"
    ):
        parser.CommandLineParser.azimuth_angle("TEST")
    with caplog.at_level("ERROR"):
        with pytest.raises(TypeError):
            parser.CommandLineParser.azimuth_angle([0, 10])
    assert "The azimuth angle provided is not a valid numerical or string value." in caplog.text


def test_initialize_default_arguments():

    # default arguments
    _parser_1 = parser.CommandLineParser()
    _parser_1.initialize_default_arguments()
    job_groups = _parser_1._action_groups
    for group in job_groups:
        assert str(group.title) in [
            "positional arguments",
            "optional arguments",
            "options",
            "paths",
            "configuration",
            "execution",
        ]

    _parser_2 = parser.CommandLineParser()
    _parser_2.initialize_default_arguments(output=True)
    job_groups = _parser_2._action_groups
    assert "output" in [str(group.title) for group in job_groups]


def test_simulation_model():
    # simulation model is none
    _parser_n = parser.CommandLineParser()
    _parser_n.initialize_default_arguments(simulation_model=None)

    # Model version only, no site or telescope
    _parser_v = parser.CommandLineParser()
    _parser_v.initialize_default_arguments(simulation_model=["model_version"])
    job_groups = _parser_v._action_groups

    assert SIMULATION_MODEL_STRING in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "model_version" for action in group._group_actions)
            assert all(action.dest != "site" for action in group._group_actions)
            assert all(action.dest != "telescope" for action in group._group_actions)

    # Site model can exist without a telescope model, model and parameter version
    _parser_s = parser.CommandLineParser()
    _parser_s.initialize_default_arguments(
        simulation_model=["site", "model_version", "parameter_version"]
    )
    job_groups = _parser_s._action_groups
    assert SIMULATION_MODEL_STRING in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "model_version" for action in group._group_actions)
            assert any(action.dest == "parameter_version" for action in group._group_actions)
            assert any(action.dest == "site" for action in group._group_actions)
            assert all(action.dest != "telescope" for action in group._group_actions)

    # No telescope model without site model; parameter_version only
    _parser_t = parser.CommandLineParser()
    _parser_t.initialize_default_arguments(
        simulation_model=["telescope", "site", "parameter_version"]
    )
    job_groups = _parser_t._action_groups
    assert SIMULATION_MODEL_STRING in [str(group.title) for group in job_groups]
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "parameter_version" for action in group._group_actions)
            assert any(action.dest == "site" for action in group._group_actions)
            assert any(action.dest == "telescope" for action in group._group_actions)


def test_job_submission():
    _parser_5 = parser.CommandLineParser()
    _parser_5.initialize_default_arguments(job_submission=True)
    job_groups = _parser_5._action_groups
    assert "job submission" in [str(group.title) for group in job_groups]


def test_db_configuration():
    _parser_6 = parser.CommandLineParser()
    _parser_6.initialize_default_arguments(db_config=True)
    job_groups = _parser_6._action_groups
    assert "database configuration" in [str(group.title) for group in job_groups]


def test_layout_parsers():
    _parser_7 = parser.CommandLineParser()
    _parser_7.initialize_default_arguments(simulation_model=["layout"])
    job_groups = _parser_7._action_groups
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "array_layout_name" for action in group._group_actions)
            assert any(action.dest == "array_element_list" for action in group._group_actions)

    _parser_8 = parser.CommandLineParser()
    _parser_8.initialize_default_arguments(simulation_model=["layout", "layout_file"])
    job_groups = _parser_8._action_groups
    for group in job_groups:
        if str(group.title) == SIMULATION_MODEL_STRING:
            assert any(action.dest == "array_layout_name" for action in group._group_actions)
            assert any(action.dest == "array_element_list" for action in group._group_actions)
            assert any(action.dest == "array_layout_file" for action in group._group_actions)


def test_simulation_configuration():
    _parser_9 = parser.CommandLineParser()
    _parser_9.initialize_default_arguments(
        simulation_configuration={"software": None, "corsika_configuration": ["all"]}
    )
    job_groups = _parser_9._action_groups
    for group in job_groups:
        if str(group.title) == "simulation software":
            assert any(action.dest == "simulation_software" for action in group._group_actions)
        if str(group.title) == "simulation configuration":
            assert any(action.dest == "primary" for action in group._group_actions)
        if str(group.title) == "shower parameters":
            assert any(action.dest == "view_cone" for action in group._group_actions)

    _parser_10 = parser.CommandLineParser()
    _parser_10.initialize_default_arguments(
        simulation_configuration={"software": None, "corsika_configuration": ["wrong_parameter"]}
    )


def test_initialize_db_config_arguments_strip_string():
    parser_10 = parser.CommandLineParser()
    parser_10.initialize_db_config_arguments()
    for test_string in ["test", " test", "test ", " test "]:
        args = parser_10.parse_args(["--db_simulation_model", test_string])
        assert args.db_simulation_model == "test"
