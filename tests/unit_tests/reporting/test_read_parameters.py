import astropy.units as u

from simtools.reporting.docs_read_parameters import ReadParameters


def test_get_all_parameter_descriptions(telescope_model_lst, io_handler):

    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(
        db_config=None, telescope_model=telescope_model_lst, output_path=output_path
    )

    # Call get_all_parameter_descriptions
    descriptions, short_descriptions, inst_class = read_parameters.get_all_parameter_descriptions()

    assert isinstance(descriptions.get("focal_length"), str)
    assert isinstance(short_descriptions.get("focal_length"), str)
    assert isinstance(inst_class.get("focal_length"), str)


def test_get_telescope_parameter_data(telescope_model_lst, io_handler):

    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(
        db_config=None, telescope_model=telescope_model_lst, output_path=output_path
    )

    result = read_parameters.get_telescope_parameter_data(telescope_model_lst)

    # Assert the result contains the expected data
    if result[1] == "focal_length":
        assert result[0] == "Structure"
        assert result[3] == (2800.0 * u.cm)
        assert result[4] == "Nominal overall focal length of the entire telescope."


def test_generate_array_element_report(telescope_model_lst, io_handler):
    output_path = io_handler.get_output_directory(sub_dir=f"{telescope_model_lst.model_version}")
    read_parameters = ReadParameters(
        db_config=None, telescope_model=telescope_model_lst, output_path=output_path
    )

    read_parameters.generate_array_element_report()

    file_path = output_path / f"{telescope_model_lst.name}.md"
    assert file_path.exists()
