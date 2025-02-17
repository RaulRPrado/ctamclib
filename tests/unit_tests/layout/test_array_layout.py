#!/usr/bin/python3

import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from simtools.data_model import data_reader
from simtools.layout.array_layout import ArrayLayout, InvalidTelescopeListFileError

logger = logging.getLogger()


@pytest.fixture
def array_layout_north_instance(io_handler, db_config, model_version):
    return ArrayLayout(
        site="North", mongo_db_config=db_config, model_version=model_version, name="test_layout"
    )


@pytest.fixture
def array_layout_south_instance(io_handler, db_config, model_version):
    return ArrayLayout(
        site="South", mongo_db_config=db_config, model_version=model_version, name="test_layout"
    )


@pytest.fixture
def north_layout_center_data_dict():
    return {
        "center_lon": -17.8920302 * u.deg,
        "center_lat": 28.7621661 * u.deg,
        "center_easting": 217611.227 * u.m,
        "center_northing": 3185066.278 * u.m,
        "EPSG": 32628,
        "center_alt": 2177 * u.m,
    }


@pytest.fixture
def south_layout_center_data_dict():
    return {
        "center_lon": -70.316345 * u.deg,
        "center_lat": -24.683429 * u.deg,
        "center_easting": 366822.017 * u.m,
        "center_northing": 7269466.999 * u.m,
        "EPSG": 32719,
        "center_alt": 2162.35 * u.m,
    }


@pytest.fixture
def array_layout_north_four_lst_instance(db_config, model_version):
    return ArrayLayout(
        site="North",
        mongo_db_config=db_config,
        label="test_layout",
        name="LST4",
        model_version=model_version,
    )


@pytest.fixture
def array_layout_south_four_lst_instance(db_config, model_version):
    return ArrayLayout(
        site="South",
        mongo_db_config=db_config,
        label="test_layout",
        name="LST4",
        model_version=model_version,
    )


def test_initialize_site_parameters_from_db():

    with pytest.raises(ValueError, match="No database configuration provided"):
        ArrayLayout(site="North", mongo_db_config=None, model_version="test_model_version")


def test_initialize_coordinate_systems(
    north_layout_center_data_dict,
    array_layout_north_instance,
    south_layout_center_data_dict,
    array_layout_south_instance,
):
    def test_one_site(center_data_dict, instance, easting, northing):
        # set center data from database
        instance._initialize_coordinate_systems()
        _x, _y, _z = instance._array_center.get_coordinates("ground")
        assert _x == 0.0 * u.m
        assert _y == 0.0 * u.m
        assert _z.value == pytest.approx(
            instance._reference_position_dict["center_altitude"].value, 1.0e-2
        )
        _lat, _lon, _z = instance._array_center.get_coordinates("mercator")
        assert _z.value == pytest.approx(
            instance._reference_position_dict["center_altitude"].value, 1.0e-2
        )
        assert _lat.value == pytest.approx(center_data_dict["center_lat"].value, 1.0e-2)
        assert _lon.value == pytest.approx(center_data_dict["center_lon"].value, 1.0e-2)

        _e, _n, _z = instance._array_center.get_coordinates("utm")
        assert _z.value == pytest.approx(
            instance._reference_position_dict["center_altitude"].value, 1.0e-2
        )
        assert _e.value == pytest.approx(
            instance._reference_position_dict["center_easting"].value, 1.0
        )
        assert _n.value == pytest.approx(
            instance._reference_position_dict["center_northing"].value, 1.0
        )

        assert instance.name == "test_layout"

    test_one_site(
        north_layout_center_data_dict, array_layout_north_instance, 217611.227, 3185066.278
    )
    test_one_site(
        south_layout_center_data_dict, array_layout_south_instance, 366822.017, 7269466.999
    )


def test_select_assets(
    telescope_north_with_calibration_devices_test_file, db_config, model_version
):
    layout = ArrayLayout(
        site="North",
        name="test_layout",
        telescope_list_file=telescope_north_with_calibration_devices_test_file,
        mongo_db_config=db_config,
        model_version=model_version,
    )

    layout.select_assets(None)
    assert len(layout._telescope_list) == 25

    layout.select_assets([])
    assert len(layout._telescope_list) == 25

    layout.select_assets(["MSTN", "SSTN"])
    assert len(layout._telescope_list) == 9

    layout.select_assets(["NOT_AN_ASSET", "ALSO_NOT_AN_ASSET"])
    assert len(layout._telescope_list) == 0


def test_add_tel(
    array_layout_north_instance,
    array_layout_south_instance,
):
    def test_one_site(instance, altitude, tel_name, design_model):
        ntel_before = instance.get_number_of_telescopes()
        instance.add_telescope(
            tel_name, "ground", 100.0 * u.m, 50.0 * u.m, 2177.0 * u.m, design_model=design_model
        )
        ntel_after = instance.get_number_of_telescopes()
        assert ntel_before + 1 == ntel_after

        instance.add_telescope(
            tel_name, "ground", 100.0 * u.m, 50.0 * u.m, None, 50.0 * u.m, design_model=design_model
        )
        assert instance._telescope_list[-1].get_altitude().value == pytest.approx(altitude)

    test_one_site(array_layout_north_instance, 2197.0, "MSTN-20", "MSTx-NectarCam")
    test_one_site(array_layout_south_instance, 2181.0, "LSTS-05", "LSTS-design")


def check_table_columns(_table, add_geocode, asset_code, sequence_number):
    """Helper function for test_build_layout to check table columns."""
    if add_geocode:
        assert "geo_code" in _table.colnames
    elif asset_code and not sequence_number:
        assert "asset_code" not in _table.colnames
    elif not asset_code and sequence_number:
        assert "sequence_number" not in _table.colnames
    elif asset_code and sequence_number:
        assert "asset_code" in _table.colnames
        assert "sequence_number" in _table.colnames


def test_build_layout(
    array_layout_north_four_lst_instance,
    array_layout_south_four_lst_instance,
    tmp_test_directory,
    db_config,
    io_handler,
    capfd,
):
    def test_one_site(
        layout, site_label, add_geocode=False, asset_code=False, sequence_number=False
    ):
        layout.add_telescope(
            telescope_name="LST" + site_label[0] + "-01",
            crs_name="ground",
            xx=57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        layout.add_telescope(
            telescope_name="LST" + site_label[0] + "-02",
            crs_name="ground",
            xx=-57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        layout.add_telescope(
            telescope_name="LST" + site_label[0] + "-02",
            crs_name="ground",
            xx=57.5 * u.m,
            yy=-57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        layout.add_telescope(
            telescope_name="LST" + site_label[0] + "-04",
            crs_name="ground",
            xx=-57.5 * u.m,
            yy=-57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )
        for tel in layout._telescope_list:
            if add_geocode:
                tel.geo_code = "test_geo_code"
            if asset_code:
                tel.asset_code = "test_asset_code"
            if sequence_number:
                tel.sequence_number = 1

        layout.convert_coordinates()
        # test that certain keywords appear in printout
        layout.print_telescope_list("ground")
        captured_printout, _ = capfd.readouterr()
        substrings_to_check = ["LST", "telescope_name", "position_x"]
        assert all(substring in captured_printout for substring in substrings_to_check)

        _table = layout.export_telescope_list_table(crs_name="ground")
        _export_file = tmp_test_directory / "test_layout.ecsv"
        _table.write(_export_file, format="ascii.ecsv", overwrite=True)

        assert isinstance(_table, QTable)

        check_table_columns(_table, add_geocode, asset_code, sequence_number)

    test_one_site(array_layout_north_four_lst_instance, "North")
    test_one_site(array_layout_south_four_lst_instance, "South")
    test_one_site(array_layout_north_four_lst_instance, "North", add_geocode=True)
    test_one_site(
        array_layout_north_four_lst_instance, "North", asset_code=True, sequence_number=False
    )
    test_one_site(
        array_layout_north_four_lst_instance, "North", asset_code=False, sequence_number=True
    )
    test_one_site(
        array_layout_north_four_lst_instance, "North", asset_code=True, sequence_number=True
    )


def test_converting_center_coordinates_north(array_layout_north_four_lst_instance):
    layout = array_layout_north_four_lst_instance

    _lat, _lon, _ = layout._array_center.get_coordinates("mercator")
    assert _lat.value == pytest.approx(28.7621661)
    assert _lon.value == pytest.approx(-17.8920302)

    _east, _north, _ = layout._array_center.get_coordinates("utm")
    assert _north.value == pytest.approx(3185066.278)
    assert _east.value == pytest.approx(217608.975)

    assert layout._array_center.get_altitude().value == pytest.approx(2177.0)


def test_converting_center_coordinates_south(array_layout_south_four_lst_instance):
    layout = array_layout_south_four_lst_instance

    _lat, _lon, _ = layout._array_center.get_coordinates("mercator")
    assert _lat.value == pytest.approx(-24.68342915473787)
    assert _lon.value == pytest.approx(-70.31634499364885)

    _east, _north, _ = layout._array_center.get_coordinates("utm")
    assert _north.value == pytest.approx(7269466.0)
    assert _east.value == pytest.approx(366822.0)

    assert layout._array_center.get_altitude().value == pytest.approx(2162.0)


def test_altitude_from_corsika_z(
    array_layout_north_four_lst_instance, array_layout_south_four_lst_instance
):
    def test_one_site(instance, telescope_name, result1, result2):
        instance.add_telescope(
            telescope_name=telescope_name,
            crs_name="ground",
            xx=57.5 * u.m,
            yy=57.5 * u.m,
            tel_corsika_z=0 * u.m,
        )

        assert instance._altitude_from_corsika_z(
            pos_z=result2 * u.m, altitude=None, telescope_axis_height=16.0 * u.m
        ).value == pytest.approx(result1)
        assert instance._altitude_from_corsika_z(
            pos_z=None, altitude=result1 * u.m, telescope_axis_height=16.0 * u.m
        ).value == pytest.approx(result2)
        with pytest.raises(TypeError):
            instance._altitude_from_corsika_z(5.0, None, telescope_axis_height=16.0 * u.m)
        assert np.isnan(instance._altitude_from_corsika_z(None, None, None))

    test_one_site(array_layout_north_four_lst_instance, "LSTN-01", 2185.0, 45.0)
    test_one_site(array_layout_south_four_lst_instance, "LSTS-01", 2176.0, 45.0)


def test_try_set_altitude(
    array_layout_north_instance,
    telescope_north_test_file,
    array_layout_south_instance,
    telescope_south_test_file,
):
    obs_level_north = 2158.0
    manual_z_positions_north = [43.00, 32.00, 28.70, 32.00, 50.3, 24.0]
    telescope_axis_height_north = [16.0, 16.0, 16.0, 16.0, 9.0, 9.0]

    obs_level_south = 2147.0
    manual_z_positions_south = [34.30, 29.40, 31.00, 33.10, 24.35, 31.00]
    telescope_axis_height_south = [16.0, 16.0, 16.0, 16.0, 9.0, 9.0]

    def test_one_site(test_file, instance, obs_level, manual_z_positions, telescope_axis_height):
        table = data_reader.read_table_from_file(test_file, validate=False)
        manual_altitudes = [
            manual_z_positions[step] + obs_level - telescope_axis_height[step] for step in range(6)
        ]
        for step, row in enumerate(table[:6]):
            tel = instance._load_telescope_names(row)
            instance._set_telescope_auxiliary_parameters(tel)
            instance._try_set_altitude(row, tel, table)
            for key, _crs in tel.crs.items():
                if key == "auxiliary":
                    continue
                assert pytest.approx(_crs["zz"]["value"], 0.01) == manual_altitudes[step]

    test_one_site(
        telescope_north_test_file,
        array_layout_north_instance,
        obs_level_north,
        manual_z_positions_north,
        telescope_axis_height_north,
    )
    test_one_site(
        telescope_south_test_file,
        array_layout_south_instance,
        obs_level_south,
        manual_z_positions_south,
        telescope_axis_height_south,
    )


def test_try_set_coordinate(
    array_layout_north_instance,
    telescope_north_test_file,
    array_layout_south_instance,
    telescope_south_test_file,
):
    manual_xx_north = [-70.99, -35.38, 75.22, 30.78, -211.61, -153.34]
    manual_yy_north = [-52.08, 66.14, 50.45, -64.51, 5.67, 169.04]

    manual_xx_south = [-20.63, 80.04, -19.39, -105.06, 0.00, 1.45]
    manual_yy_south = [-64.84, -0.77, 65.22, 1.15, 0.00, 151.07]

    def test_one_site(instance, test_file, manual_xx, manual_yy):
        table = data_reader.read_table_from_file(test_file, validate=False)
        for step, row in enumerate(table[:6]):
            tel = instance._load_telescope_names(row)
            instance._try_set_coordinate(row, tel, table, "ground", "position_x", "position_y")
            assert pytest.approx(tel.crs["ground"]["xx"]["value"], 0.1) == manual_xx[step]
            assert pytest.approx(tel.crs["ground"]["yy"]["value"], 0.1) == manual_yy[step]

        del table["telescope_name"]
        for step, row in enumerate(table[:6]):
            with pytest.raises(
                InvalidTelescopeListFileError,
                match="Missing required row with telescope_name or asset_code/sequence_number",
            ):
                instance._load_telescope_names(row)

    test_one_site(
        array_layout_north_instance, telescope_north_test_file, manual_xx_north, manual_yy_north
    )
    test_one_site(
        array_layout_south_instance, telescope_south_test_file, manual_xx_south, manual_yy_south
    )


def test_len(telescope_north_test_file, db_config, model_version):
    layout = ArrayLayout(
        telescope_list_file=telescope_north_test_file,
        mongo_db_config=db_config,
        model_version=model_version,
        site="North",
    )
    assert len(layout) == 13
    assert layout.get_number_of_telescopes() == 13


def test_getitem(db_config, telescope_north_test_file, model_version):
    layout = ArrayLayout(
        telescope_list_file=telescope_north_test_file,
        mongo_db_config=db_config,
        model_version=model_version,
        site="North",
    )

    assert layout[0].name == "LSTN-01"


def test_export_telescope_list_table(
    db_config, telescope_north_test_file, telescope_north_utm_test_file, model_version
):
    layout = ArrayLayout(
        mongo_db_config=db_config,
        site="North",
        model_version=model_version,
        telescope_list_file=telescope_north_test_file,
    )
    table = layout.export_telescope_list_table(crs_name="ground")
    assert isinstance(table, QTable)

    assert "telescope_name" in table.colnames
    assert "geo_code" in table.colnames
    assert "position_z" in table.colnames
    assert "sequence_number" not in table.colnames

    layout_utm = ArrayLayout(
        mongo_db_config=db_config,
        site="North",
        model_version=model_version,
        telescope_list_file=telescope_north_utm_test_file,
    )
    table_utm = layout_utm.export_telescope_list_table(crs_name="utm")
    assert "asset_code" in table_utm.colnames
    assert "sequence_number" in table_utm.colnames
    assert "geo_code" in table_utm.colnames
    assert "telescope_name" not in table_utm.colnames

    layout_utm._telescope_list = []
    try:
        table_utm = layout_utm.export_telescope_list_table(crs_name="utm")
    except IndexError:
        pytest.fail("IndexError raised")


def test_export_one_telescope_as_json(db_config, model_version, telescope_north_utm_test_file):

    layout = ArrayLayout(
        mongo_db_config=db_config,
        site="North",
        model_version=model_version,
        telescope_list_file="tests/resources/model_parameters/array_element_position_ground-2.0.0.json",
    )

    ground_dict = layout.export_one_telescope_as_json(crs_name="ground")
    assert isinstance(ground_dict, dict)
    assert ground_dict["instrument"] == "MSTN-09"
    assert ground_dict["parameter"] == "array_element_position_ground"

    utm_dict = layout.export_one_telescope_as_json(crs_name="utm")
    assert utm_dict["parameter"] == "array_element_position_utm"

    mercator_dict = layout.export_one_telescope_as_json(crs_name="mercator")
    assert mercator_dict["parameter"] == "array_element_position_mercator"

    layout_utm = ArrayLayout(
        mongo_db_config=db_config,
        site="North",
        model_version=model_version,
        telescope_list_file=telescope_north_utm_test_file,
    )
    with pytest.raises(ValueError, match=r"Only one telescope can be exported to json"):
        layout_utm.export_one_telescope_as_json(crs_name="ground")


def test_read_table_from_json_file(db_config, model_version):

    ground_table_file = "tests/resources/model_parameters/array_element_position_ground-2.0.0.json"
    layout = ArrayLayout(
        mongo_db_config=db_config,
        site="North",
        model_version=model_version,
        telescope_list_file=ground_table_file,
    )
    ground_table = layout._read_table_from_json_file(ground_table_file)
    assert isinstance(ground_table, QTable)
    assert "position_x" in ground_table.colnames

    utm_table = layout._read_table_from_json_file(
        "tests/resources/model_parameters/array_element_position_utm-2.0.0.json"
    )
    assert isinstance(utm_table, QTable)
    assert "utm_north" in utm_table.colnames
