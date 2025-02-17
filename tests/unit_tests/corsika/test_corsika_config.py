#!/usr/bin/python3

import logging
import pathlib
from unittest.mock import Mock, patch

import pytest

from simtools.corsika.corsika_config import CorsikaConfig

logger = logging.getLogger()


@pytest.fixture
def corsika_config_no_array_model(corsika_config_data):
    corsika_config_data["correct_for_b_field_alignment"] = False
    return CorsikaConfig(
        array_model=None, label="test-corsika-config", args_dict=corsika_config_data, db_config=None
    )


@pytest.fixture
def gcm2():
    return "g/cm2"


@pytest.fixture
def corsika_configuration_parameters(gcm2):
    return {
        "corsika_iact_max_bunches": {"value": 1000000, "unit": None},
        "corsika_cherenkov_photon_bunch_size": {"value": 5.0, "unit": None},
        "corsika_cherenkov_photon_wavelength_range": {"value": [240.0, 1000.0], "unit": "nm"},
        "corsika_first_interaction_height": {"value": 0.0, "unit": "cm"},
        "corsika_particle_kinetic_energy_cutoff": {
            "value": [0.3, 0.1, 0.020, 0.020],
            "unit": "GeV",
        },
        "corsika_longitudinal_shower_development": {"value": 20.0, "unit": gcm2},
        "corsika_iact_split_auto": {"value": 15000000, "unit": None},
        "corsika_starting_grammage": {"value": 0.0, "unit": gcm2},
        "corsika_iact_io_buffer": {"value": 800, "unit": "MB"},
    }


def test_repr(corsika_config_mock_array_model):
    assert "site" in repr(corsika_config_mock_array_model)


def test_fill_corsika_configuration(io_handler, corsika_config_mock_array_model):

    empty_config = CorsikaConfig(
        array_model=None, label="test-corsika-config", args_dict=None, db_config=None
    )
    assert empty_config.config == {}

    assert corsika_config_mock_array_model.get_config_parameter("NSHOW") == 100
    assert corsika_config_mock_array_model.get_config_parameter("THETAP") == [20, 20]
    assert corsika_config_mock_array_model.get_config_parameter("ERANGE") == [10.0, 10000.0]
    # Testing conversion between AZM (sim_telarray) and PHIP (corsika)
    assert corsika_config_mock_array_model.get_config_parameter("PHIP") == [175.467, 175.467]
    assert corsika_config_mock_array_model.get_config_parameter("CSCAT") == [10, 140000.0, 0]

    # db_config is not None
    assert pytest.approx(corsika_config_mock_array_model.get_config_parameter("CERSIZ")) == 5.0
    assert corsika_config_mock_array_model.get_config_parameter("MAX_BUNCHES") == 1000000
    assert corsika_config_mock_array_model.get_config_parameter("ECUTS") == "0.3 0.1 0.02 0.02"

    for key in [
        "USER_INPUT",
        "INTERACTION_FLAGS",
        "CHERENKOV_EMISSION_PARAMETERS",
        "DEBUGGING_OUTPUT_PARAMETERS",
        "IACT_PARAMETERS",
    ]:
        assert key in corsika_config_mock_array_model.config


def test_corsika_configuration_from_user_input(
    corsika_config_mock_array_model, corsika_config_data
):
    user_dict = corsika_config_mock_array_model._corsika_configuration_from_user_input(
        corsika_config_data
    )
    assert isinstance(user_dict, dict)
    assert "CSCAT" in user_dict
    assert isinstance(user_dict["CSCAT"], list)


def test_corsika_configuration_interaction_flags(
    corsika_config_mock_array_model, corsika_configuration_parameters
):
    with pytest.raises(KeyError):
        corsika_config_mock_array_model._corsika_configuration_interaction_flags({})

    parameters = corsika_config_mock_array_model._corsika_configuration_interaction_flags(
        corsika_configuration_parameters
    )
    assert isinstance(parameters, dict)
    assert "ECUTS" in parameters
    assert parameters["MAXPRT"] == ["10"]
    assert len(parameters) == 9


def test_input_config_first_interaction_height(corsika_config_mock_array_model):
    assert corsika_config_mock_array_model._input_config_first_interaction_height(
        {"value": 0.0, "unit": "cm"}
    ) == ["0.00", "0"]

    assert corsika_config_mock_array_model._input_config_first_interaction_height(
        {"value": 10.0, "unit": "m"}
    ) == ["1000.00", "0"]


def test_input_config_corsika_starting_grammage(corsika_config_mock_array_model, gcm2):
    assert (
        corsika_config_mock_array_model._input_config_corsika_starting_grammage(
            {"value": 0.0, "unit": gcm2}
        )
        == "0.0"
    )
    assert (
        corsika_config_mock_array_model._input_config_corsika_starting_grammage(
            {"value": 10.0, "unit": "kg/cm2"}
        )
        == "10000.0"
    )


def test_input_config_corsika_particle_kinetic_energy_cutoff(corsika_config_mock_array_model):
    assert corsika_config_mock_array_model._input_config_corsika_particle_kinetic_energy_cutoff(
        {"value": [0.3, 0.1, 0.020, 0.020], "unit": "GeV"}
    ) == ["0.3 0.1 0.02 0.02"]
    assert corsika_config_mock_array_model._input_config_corsika_particle_kinetic_energy_cutoff(
        {"value": [0.3, 0.1, 0.020, 0.020], "unit": "TeV"}
    ) == ["300.0 100.0 20.0 20.0"]


def test_input_config_corsika_longitudinal_parameters(corsika_config_mock_array_model, gcm2):
    assert corsika_config_mock_array_model._input_config_corsika_longitudinal_parameters(
        {"value": 20.0, "unit": gcm2}
    ) == ["T", "20.0", "F", "F"]
    assert corsika_config_mock_array_model._input_config_corsika_longitudinal_parameters(
        {"value": 10.0, "unit": "kg/cm2"}
    ) == ["T", "10000.0", "F", "F"]


def test_corsika_configuration_cherenkov_parameters(
    corsika_config_mock_array_model, corsika_configuration_parameters
):
    cherenk_dict = corsika_config_mock_array_model._corsika_configuration_cherenkov_parameters(
        corsika_configuration_parameters
    )
    assert isinstance(cherenk_dict, dict)
    assert "CERSIZ" in cherenk_dict
    assert isinstance(cherenk_dict["CERSIZ"], list)


def test_input_config_corsika_cherenkov_wavelength(corsika_config_mock_array_model):
    assert corsika_config_mock_array_model._input_config_corsika_cherenkov_wavelength(
        {"value": [240.0, 1000.0], "unit": "nm"}
    ) == ["240.0", "1000.0"]


def test_corsika_configuration_iact_parameters(
    corsika_config_mock_array_model, corsika_configuration_parameters
):
    iact_dict = corsika_config_mock_array_model._corsika_configuration_iact_parameters(
        corsika_configuration_parameters
    )
    assert isinstance(iact_dict, dict)
    assert "MAX_BUNCHES" in iact_dict
    assert isinstance(iact_dict["MAX_BUNCHES"], list)


def test_input_config_io_buff(corsika_config_mock_array_model):
    assert (
        corsika_config_mock_array_model._input_config_io_buff({"value": 800, "unit": "MB"})
        == "800MB"
    )
    assert (
        corsika_config_mock_array_model._input_config_io_buff({"value": 8.5, "unit": "MB"})
        == "8500000"
    )
    assert (
        corsika_config_mock_array_model._input_config_io_buff({"value": 800, "unit": "kB"})
        == "800000"
    )


def test_corsika_configuration_debugging_parameters(corsika_config_mock_array_model):
    assert isinstance(
        corsika_config_mock_array_model._corsika_configuration_debugging_parameters(), dict
    )
    assert len(corsika_config_mock_array_model._corsika_configuration_debugging_parameters()) == 4


def test_rotate_azimuth_by_180deg_no_correct_for_geomagnetic_field_alignment(
    corsika_config_mock_array_model,
):
    assert (
        pytest.approx(
            corsika_config_mock_array_model._rotate_azimuth_by_180deg(
                0.0, correct_for_geomagnetic_field_alignment=False
            )
        )
        == 180.0
    )
    assert (
        pytest.approx(
            corsika_config_mock_array_model._rotate_azimuth_by_180deg(
                360.0, correct_for_geomagnetic_field_alignment=False
            )
        )
        == 180.0
    )
    assert (
        pytest.approx(
            corsika_config_mock_array_model._rotate_azimuth_by_180deg(
                450.0, correct_for_geomagnetic_field_alignment=False
            )
        )
        == 270.0
    )
    assert (
        pytest.approx(
            corsika_config_mock_array_model._rotate_azimuth_by_180deg(
                180.0, correct_for_geomagnetic_field_alignment=False
            )
        )
        == 0.0
    )
    assert (
        pytest.approx(
            corsika_config_mock_array_model._rotate_azimuth_by_180deg(
                -180.0, correct_for_geomagnetic_field_alignment=False
            )
        )
        == 0.0
    )


def test_rotate_azimuth_by_180deg(corsika_config_mock_array_model):
    assert (
        pytest.approx(
            corsika_config_mock_array_model._rotate_azimuth_by_180deg(
                0.0, correct_for_geomagnetic_field_alignment=True
            )
        )
        == 175.467
    )
    assert (
        pytest.approx(
            corsika_config_mock_array_model._rotate_azimuth_by_180deg(
                360.0, correct_for_geomagnetic_field_alignment=True
            )
        )
        == 175.467
    )
    assert (
        pytest.approx(
            corsika_config_mock_array_model._rotate_azimuth_by_180deg(
                450.0, correct_for_geomagnetic_field_alignment=True
            )
        )
        == 265.467
    )
    assert (
        pytest.approx(
            corsika_config_mock_array_model._rotate_azimuth_by_180deg(
                180.0, correct_for_geomagnetic_field_alignment=True
            )
        )
        == 355.467
    )
    assert (
        pytest.approx(
            corsika_config_mock_array_model._rotate_azimuth_by_180deg(
                -180.0, correct_for_geomagnetic_field_alignment=True
            )
        )
        == 355.467
    )


def test_set_primary_particle(corsika_config_mock_array_model):
    from simtools.corsika.primary_particle import PrimaryParticle

    cc = corsika_config_mock_array_model
    assert isinstance(cc._set_primary_particle(args_dict=None), PrimaryParticle)
    assert isinstance(
        cc._set_primary_particle(args_dict={"primary_id_type": None}), PrimaryParticle
    )

    p_common_name = cc._set_primary_particle(
        args_dict={"primary": "proton", "primary_id_type": "common_name"}
    )
    assert p_common_name.name == "proton"

    p_corsika7_id = cc._set_primary_particle(
        args_dict={"primary": 14, "primary_id_type": "corsika7_id"}
    )
    assert p_corsika7_id.name == "proton"

    p_pdg_id = cc._set_primary_particle(args_dict={"primary": 2212, "primary_id_type": "pdg_id"})
    assert p_pdg_id.name == "proton"


def test_get_config_parameter(corsika_config_mock_array_model, caplog):
    cc = corsika_config_mock_array_model
    assert isinstance(cc.get_config_parameter("NSHOW"), int)
    assert isinstance(cc.get_config_parameter("THETAP"), list)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyError):
            cc.get_config_parameter("not_really_a_parameter")
    assert "Parameter not_really_a_parameter" in caplog.text


def test_print_config_parameter(corsika_config_mock_array_model, capsys):
    logger.info("test_print_config_parameter")
    corsika_config_mock_array_model.print_config_parameter()
    assert "NSHOW" in capsys.readouterr().out


def test_get_text_single_line(corsika_config_mock_array_model):
    assert (
        corsika_config_mock_array_model._get_text_single_line({"EVTNR": [1], "RUNNR": [10]})
        == "EVTNR 1 \nRUNNR 10 \n"
    )
    assert (
        corsika_config_mock_array_model._get_text_single_line(
            {"SPLIT_AUTO": ["15M"], "IO_BUFFER": ["800MB"], "MAX_BUNCHES": ["1000000"]}, "IACT "
        )
        == "IACT SPLIT_AUTO 15M \nIACT IO_BUFFER 800MB \nIACT MAX_BUNCHES 1000000 \n"
    )


def test_generate_corsika_input_file(corsika_config_mock_array_model):
    logger.info("test_generate_corsika_input_file")
    input_file = corsika_config_mock_array_model.generate_corsika_input_file()
    assert input_file.exists()
    with open(input_file) as f:
        assert "TELFIL |" not in f.read()

    assert corsika_config_mock_array_model._is_file_updated
    assert input_file == corsika_config_mock_array_model.generate_corsika_input_file()


def test_generate_corsika_input_file_multipipe(corsika_config_mock_array_model):
    logger.info("test_generate_corsika_input_file")
    input_file = corsika_config_mock_array_model.generate_corsika_input_file(use_multipipe=True)
    assert input_file.exists()
    with open(input_file) as f:
        assert "TELFIL |" in f.read()


def test_generate_corsika_input_file_with_test_seeds(corsika_config_mock_array_model):
    logger.info("test_generate_corsika_input_file_with_test_seeds")
    input_file = corsika_config_mock_array_model.generate_corsika_input_file(use_test_seeds=True)
    assert input_file.exists()
    expected_seeds = [534, 220, 1104, 382]
    with open(input_file) as f:
        file_content = f.read()
        for seed in expected_seeds:
            assert f"SEED {seed} 0 0" in file_content


def test_get_corsika_config_file_name(corsika_config_mock_array_model, io_handler):
    file_name = "proton_South_test_layout_za020-azm000deg_cone0-10_test-corsika-config"

    assert (
        corsika_config_mock_array_model.get_corsika_config_file_name("config_tmp", run_number=1)
        == f"corsika_config_run000001_{file_name}.txt"
    )
    with pytest.raises(
        ValueError, match="Must provide a run number for a temporary CORSIKA config file"
    ):
        assert (
            corsika_config_mock_array_model.get_corsika_config_file_name("config_tmp")
            == f"corsika_config_run000001_{file_name}.txt"
        )

    assert (
        corsika_config_mock_array_model.get_corsika_config_file_name("config")
        == f"corsika_config_{file_name}.input"
    )
    # The test below includes the placeholder XXXXXX for the run number because
    # that is the way we get the run number later in the CORSIKA input file with zero padding.
    assert corsika_config_mock_array_model.get_corsika_config_file_name("output_generic") == (
        "runXXXXXX_proton_South_test_layout_za020-azm000deg_cone0-10_test"
        "-corsika-config_South_test_layout_test-corsika-config.zst"
    )
    assert (
        corsika_config_mock_array_model.get_corsika_config_file_name("multipipe")
        == "multi_cta-South-test_layout.cfg"
    )
    with pytest.raises(ValueError, match=r"^The requested file type"):
        corsika_config_mock_array_model.get_corsika_config_file_name("foobar")


def test_set_output_file_and_directory(corsika_config_mock_array_model):
    cc = corsika_config_mock_array_model
    output_file = cc.set_output_file_and_directory()
    assert str(output_file) == (
        "runXXXXXX_proton_South_test_layout_za020-azm000deg_cone0-10_test"
        "-corsika-config_South_test_layout_test-corsika-config.zst"
    )
    assert isinstance(cc.config_file_path, pathlib.Path)


def test_write_seeds(corsika_config_mock_array_model):
    mock_file = Mock()
    corsika_config_mock_array_model.run_number = 10
    corsika_config_mock_array_model.config = {"USER_INPUT": {"PRMPAR": [14]}}
    with patch("io.open", return_value=mock_file):
        corsika_config_mock_array_model._write_seeds(mock_file)
    assert mock_file.write.call_count == 4

    expected_calls = [_call.args[0] for _call in mock_file.write.call_args_list]
    for _call in expected_calls:
        assert _call.startswith("SEED ")
        assert _call.endswith(" 0 0\n")


def test_write_seeds_use_test_seeds(corsika_config_mock_array_model):
    mock_file = Mock()
    corsika_config_mock_array_model.run_number = 10
    corsika_config_mock_array_model.config = {"USER_INPUT": {"PRMPAR": [14]}}
    with patch("io.open", return_value=mock_file):
        corsika_config_mock_array_model._write_seeds(mock_file, use_test_seeds=True)
    assert mock_file.write.call_count == 4

    expected_calls = [_call.args[0] for _call in mock_file.write.call_args_list]
    expected_seeds = [534, 220, 1104, 382]
    for _call in expected_calls:
        assert _call == (f"SEED {expected_seeds.pop(0)} 0 0\n")


@pytest.mark.uses_model_database
def test_get_corsika_telescope_list(corsika_config):
    cc = corsika_config
    telescope_list_str = cc.get_corsika_telescope_list()
    assert telescope_list_str.count("TELESCOPE") > 0
    assert telescope_list_str.count("LSTS") > 0


def test_run_number(corsika_config_no_array_model):
    assert corsika_config_no_array_model.run_number is None
    corsika_config_no_array_model.run_number = 25
    assert corsika_config_no_array_model.run_number == 25


def test_validate_run_number(corsika_config_no_array_model):
    assert corsika_config_no_array_model.validate_run_number(None) is None
    assert corsika_config_no_array_model.validate_run_number(1)
    assert corsika_config_no_array_model.validate_run_number(123456)
    with pytest.raises(ValueError, match=r"^could not convert string to float"):
        corsika_config_no_array_model.validate_run_number("test")
    invalid_run_number = r"^Invalid type of run number"
    with pytest.raises(ValueError, match=invalid_run_number):
        corsika_config_no_array_model.validate_run_number(1.5)
    with pytest.raises(ValueError, match=invalid_run_number):
        corsika_config_no_array_model.validate_run_number(-1)
    with pytest.raises(ValueError, match=invalid_run_number):
        corsika_config_no_array_model.validate_run_number(123456789)
