#!/usr/bin/python3

import logging
from unittest import mock

import pytest

from simtools.utils import names

logging.getLogger().setLevel(logging.DEBUG)


ecsv_suffix = ".ecsv"


@pytest.fixture
def invalid_name():
    return "Invalid name"


def test_get_list_of_array_element_types():
    assert names.get_list_of_array_element_types(array_element_class="telescopes", site=None) == [
        "LSTN",
        "LSTS",
        "MSTN",
        "MSTS",
        "MSTx",
        "SCTS",
        "SSTS",
    ]

    assert names.get_list_of_array_element_types(
        array_element_class="telescopes", site="North"
    ) == [
        "LSTN",
        "MSTN",
    ]

    assert names.get_list_of_array_element_types(
        array_element_class="telescopes", site="South"
    ) == [
        "LSTS",
        "MSTS",
        "SCTS",
        "SSTS",
    ]

    assert "ILLN" in names.get_list_of_array_element_types(
        array_element_class="calibration_devices", site="North"
    )


def test_validate_name():
    with_lists = {
        "South": ["south"],
        "North": ["north"],
    }

    for key, value in with_lists.items():
        for _site in value:
            assert key == names._validate_name(_site, with_lists)

    with pytest.raises(ValueError, match=r"Invalid name Aar"):
        names._validate_name("Aar", with_lists)

    with_lists_in_dicts = {
        "LSTN": ["LSTN", "lstn"],
        "MSTS": ["MSTN", "mstn"],
    }
    for key, value in with_lists_in_dicts.items():
        for _tel in value:
            assert key == names._validate_name(_tel, with_lists_in_dicts)


def test_validate_array_element_id_name(caplog):
    _test_ids = {
        "1": "01",
        "01": "01",
        "5": "05",
        "55": "55",
        "design": "design",
        "test": "test",
    }
    for key, value in _test_ids.items():
        assert value == names.validate_array_element_id_name(key)

    assert "01" == names.validate_array_element_id_name(1)
    assert "11" == names.validate_array_element_id_name(11)

    for _id in ["no_id", "D2345", "FlashCam"]:
        with pytest.raises(ValueError, match=r"^Invalid array element ID name"):
            names.validate_array_element_id_name(_id)

    assert "FlashCam" == names.validate_array_element_id_name("FlashCam", "MSTx")


def test_array_element_design_types():
    assert names.array_element_design_types(None) == ["design", "test"]
    assert names.array_element_design_types("LSTN") == ["design", "test"]
    _expected_mstx_types = ["design", "test", "FlashCam", "NectarCam"]
    for _type in _expected_mstx_types:
        assert _type in names.array_element_design_types("MSTx")


def test_validate_site_name(invalid_name):
    for key, value in names.site_names().items():
        for test_name in value:
            assert key == names.validate_site_name(test_name)
    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.validate_site_name("Not a site")


def test_validate_array_element_type(invalid_name):
    assert "LSTN" == names.validate_array_element_type("LSTN")
    assert "ILLS" == names.validate_array_element_type("ILLS")

    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.validate_array_element_type("Not a type")
    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.validate_array_element_type("LSTN-01")


def test_validate_array_element_name(invalid_name):
    telescopes = {
        "LSTN-Design": "LSTN-Design",
        "LSTN-TEST": "LSTN-TEST",
        "LSTN-01": "LSTN-01",
        "SSTS-01": "SSTS-01",
        "OBS-North": "North",
        "MSTx-NectarCam": "MSTx-NectarCam",
    }

    for key, value in telescopes.items():
        logging.getLogger().info(f"Validating {key}")
        new_name = names.validate_array_element_name(key)
        logging.getLogger().info(f"New name {new_name}")
        assert value == new_name

    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.validate_array_element_name("South-MST-FlashCam-D")
    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.validate_array_element_name("LSTN")


def test_get_array_element_name_from_type_site_id():
    assert "LSTN-01" == names.get_array_element_name_from_type_site_id("LST", "North", "01")
    assert "LSTN-01" == names.get_array_element_name_from_type_site_id("LST", "North", "1")
    assert "LSTS-01" == names.get_array_element_name_from_type_site_id("LST", "South", "01")


def test_get_design_model_from_name():
    assert "LSTN-design" == names.get_design_model_from_name("LSTN-design")
    assert "LSTN-design" == names.get_design_model_from_name("LSTN-01")
    assert "MSTS-design" == names.get_design_model_from_name("MSTS-01")


def test_get_site_from_array_element_name(invalid_name):
    assert "North" == names.get_site_from_array_element_name("MSTN")
    assert "North" == names.get_site_from_array_element_name("MSTN-05")
    assert "South" == names.get_site_from_array_element_name("MSTS-05")
    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.get_site_from_array_element_name("LSTW")
    assert ["North", "South"] == names.get_site_from_array_element_name("MSTx")


def test_get_class_from_telescope_name(invalid_name):
    assert "telescopes" == names.get_collection_name_from_array_element_name("LSTN-01")
    assert "calibration_devices" == names.get_collection_name_from_array_element_name("ILLS-01")
    with pytest.raises(ValueError, match=rf"^{invalid_name}"):
        names.get_site_from_array_element_name("SATW")
    assert "sites" == names.get_collection_name_from_array_element_name("North")
    with pytest.raises(ValueError, match="Invalid array element name Not_a_collection"):
        names.get_collection_name_from_array_element_name("Not_a_collection")


def test_get_collection_name_from_array_element_name():

    assert "telescopes" == names.get_collection_name_from_array_element_name("LSTN-01")
    assert "telescopes" == names.get_collection_name_from_array_element_name("MSTx-FlashCam")
    assert "sites" == names.get_collection_name_from_array_element_name("OBS-North")
    assert "sites" == names.get_collection_name_from_array_element_name("North")
    assert "sites" == names.get_collection_name_from_array_element_name("OBS-North", False)
    assert "configuration_sim_telarray" == names.get_collection_name_from_array_element_name(
        "configuration_sim_telarray", False
    )

    with pytest.raises(ValueError, match=r"Invalid array element name configuration_sim_telarray"):
        names.get_collection_name_from_array_element_name("configuration_sim_telarray", True)

    with pytest.raises(ValueError, match=r"Invalid array element name Not_a_collection"):
        names.get_collection_name_from_array_element_name("Not_a_collection", False)


def test_sanitize_name(caplog):
    assert names.sanitize_name("y_edges unit") == "y_edges_unit"
    assert names.sanitize_name("Y_EDGES UNIT") == "y_edges_unit"
    assert names.sanitize_name("123name") == "_123name"
    assert names.sanitize_name("na!@#$%^&*()me") == "na__________me"
    assert names.sanitize_name("!validName") == "_validname"

    with pytest.raises(ValueError, match=r"^The string  could not be sanitized."):
        names.sanitize_name("")


def test_get_array_element_type_from_name(invalid_name):
    assert names.get_array_element_type_from_name("LSTN-01") == "LSTN"
    assert names.get_array_element_type_from_name("MSTN-02") == "MSTN"
    assert names.get_array_element_type_from_name("SSTS-27") == "SSTS"
    assert names.get_array_element_type_from_name("SCTS-27") == "SCTS"
    assert names.get_array_element_type_from_name("MAGIC-2") == "MAGIC"
    assert names.get_array_element_type_from_name("VERITAS-4") == "VERITAS"
    for _name in ["", "01", "Not_a_telescope", "LST", "MST"]:
        with pytest.raises(ValueError, match=rf"^{invalid_name}"):
            names.get_array_element_type_from_name(_name)


def test_generate_file_name_camera_efficiency():

    site = "South"
    telescope_model_name = "LSTS-01"
    zenith_angle = 20
    azimuth_angle = 180
    label = "test"

    assert (
        names.generate_file_name(
            "camera-efficiency-table",
            ecsv_suffix,
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera-efficiency-table-South-LSTS-01-za20.0deg_azm180deg_test.ecsv"
    )

    assert (
        names.generate_file_name(
            "camera-efficiency",
            ".dat",
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera-efficiency-South-LSTS-01-za20.0deg_azm180deg_test.dat"
    )

    assert (
        names.generate_file_name(
            "camera-efficiency",
            ".log",
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera-efficiency-South-LSTS-01-za20.0deg_azm180deg_test.log"
    )

    site = "North"
    telescope_model_name = "MSTN"
    zenith_angle = 40
    azimuth_angle = 0
    label = "test"
    assert (
        names.generate_file_name(
            "camera-efficiency-table",
            ecsv_suffix,
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera-efficiency-table-North-MSTN-za40.0deg_azm000deg_test.ecsv"
    )
    assert (
        names.generate_file_name(
            "camera-efficiency",
            ".dat",
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera-efficiency-North-MSTN-za40.0deg_azm000deg_test.dat"
    )
    assert (
        names.generate_file_name(
            "camera-efficiency",
            ".log",
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera-efficiency-North-MSTN-za40.0deg_azm000deg_test.log"
    )

    site = "South"
    telescope_model_name = "LSTS-01"
    zenith_angle = 20
    azimuth_angle = 180
    label = None
    assert (
        names.generate_file_name(
            "camera-efficiency-table",
            ecsv_suffix,
            site,
            telescope_model_name,
            zenith_angle,
            azimuth_angle,
            label=label,
        )
        == "camera-efficiency-table-South-LSTS-01-za20.0deg_azm180deg.ecsv"
    )


def test_simtel_telescope_config_file_name(model_version):
    assert (
        names.simtel_config_file_name(
            "South", model_version, telescope_model_name="LSTS-01", label=None, extra_label=None
        )
        == "CTA-South-LSTS-01-" + model_version + ".cfg"
    )
    assert (
        names.simtel_config_file_name(
            "South", model_version, telescope_model_name="LSTS-01", label="A", extra_label=None
        )
        == "CTA-South-LSTS-01-" + model_version + "_A.cfg"
    )
    assert (
        names.simtel_config_file_name(
            "South", model_version, telescope_model_name="LSTS-01", label="A", extra_label="B"
        )
        == "CTA-South-LSTS-01-" + model_version + "_A_B.cfg"
    )


def test_simtel_array_config_file_name(model_version):
    assert (
        names.simtel_config_file_name(
            array_name="4LSTs", site="South", model_version=model_version, label=None
        )
        == "CTA-4LSTs-South-" + model_version + ".cfg"
    )
    assert (
        names.simtel_config_file_name(
            array_name="4LSTs", site="South", model_version=model_version, label="A"
        )
        == "CTA-4LSTs-South-" + model_version + "_A.cfg"
    )


def test_simtel_single_mirror_list_file_name(model_version):
    assert (
        names.simtel_single_mirror_list_file_name(
            site="South",
            telescope_model_name="LST-1",
            model_version=model_version,
            mirror_number=5,
            label=None,
        )
        == "CTA-single-mirror-list-South-LST-1-" + model_version + "-mirror5.dat"
    )
    assert (
        names.simtel_single_mirror_list_file_name(
            site="South",
            telescope_model_name="LST-1",
            model_version=model_version,
            mirror_number=5,
            label="A",
        )
        == "CTA-single-mirror-list-South-LST-1-" + model_version + "-mirror5_A.dat"
    )


def test_layout_telescope_list_file_name():
    assert (
        names.layout_telescope_list_file_name(
            name="Alpha",
            label=None,
        )
        == "telescope_positions-Alpha.ecsv"
    )
    assert (
        names.layout_telescope_list_file_name(
            name="Alpha",
            label="sub_MST",
        )
        == "telescope_positions-Alpha_sub_MST.ecsv"
    )


def test_generate_file_name_ray_tracing():
    assert (
        names.generate_file_name(
            file_type="Photons",
            suffix=".lis",
            site="South",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            off_axis_angle=2.5,
            mirror_number=3,
            label="instance1",
        )
        == "Photons-South-LSTS-01-d10.5km-za45.0deg-off2.500deg_mirror3_instance1.lis"
    )

    assert (
        names.generate_file_name(
            file_type="log",
            site="South",
            suffix=".log",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            off_axis_angle=2.5,
            mirror_number=None,
            label=None,
        )
        == "log-South-LSTS-01-d10.5km-za45.0deg-off2.500deg.log"
    )

    assert (
        names.generate_file_name(
            file_type="ray-tracing",
            suffix=ecsv_suffix,
            site="South",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            label="instance1",
        )
        == "ray-tracing-South-LSTS-01-d10.5km-za45.0deg_instance1.ecsv"
    )
    assert (
        names.generate_file_name(
            file_type="ray-tracing",
            suffix=ecsv_suffix,
            site="South",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            label=None,
        )
        == "ray-tracing-South-LSTS-01-d10.5km-za45.0deg.ecsv"
    )

    assert (
        names.generate_file_name(
            file_type="ray-tracing",
            suffix=".pdf",
            extra_label="d80_cm",
            site="South",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            label="instance1",
        )
        == "ray-tracing-South-LSTS-01-d10.5km-za45.0deg_instance1_d80_cm.pdf"
    )
    assert (
        names.generate_file_name(
            file_type="ray-tracing",
            suffix=".pdf",
            extra_label="d80_cm",
            site="South",
            telescope_model_name="LSTS-01",
            source_distance=10.5,
            zenith_angle=45.0,
            label=None,
        )
        == "ray-tracing-South-LSTS-01-d10.5km-za45.0deg_d80_cm.pdf"
    )


def test_get_simulation_software_name_from_parameter_name():
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "focal_length", simulation_software="sim_telarray"
        )
        == "focal_length"
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "telescope_axis_height", simulation_software="sim_telarray"
        )
        is None
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "corsika_observation_level", simulation_software="sim_telarray"
        )
        == "altitude"
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "corsika_observation_level", simulation_software="corsika"
        )
        == "OBSLEV"
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "reference_point_longitude", simulation_software="sim_telarray"
        )
        is None  # this is not a sim_telarray parameter
    )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "reference_point_longitude", simulation_software="corsika"
        )
        == "reference_point_longitude"
    )
    with pytest.raises(KeyError, match="Parameter not_a_parameter without schema definition"):
        names.get_simulation_software_name_from_parameter_name(
            "not_a_parameter", simulation_software="corsika"
        )
    assert (
        names.get_simulation_software_name_from_parameter_name(
            "reference_point_longitude", simulation_software=None
        )
        is None
    )
    # Test with parameter having simulation_software as None
    with mock.patch(
        "simtools.utils.names.telescope_parameters",
        return_value={"param_with_none": {"simulation_software": None}},
    ):
        assert (
            names.get_simulation_software_name_from_parameter_name(
                "param_with_none", simulation_software="sim_telarray"
            )
            is None
        )


def test_get_parameter_name_from_simtel_name():
    assert names.get_parameter_name_from_simtel_name("focal_length") == "focal_length"
    assert names.get_parameter_name_from_simtel_name("altitude") == "corsika_observation_level"
