import astropy.units as u

from simtools.reporting.docs_read_parameters import ReadParameters


def test_get_all_parameter_descriptions(telescope_model_lst):

    read_parameters = ReadParameters(telescope_model=telescope_model_lst)

    # Call get_all_parameter_descriptions
    descriptions, short_descriptions, inst_class = read_parameters.get_all_parameter_descriptions()

    assert isinstance(descriptions.get("focal_length"), str)
    assert isinstance(short_descriptions.get("focal_length"), str)
    assert isinstance(inst_class.get("focal_length"), str)


def test_get_telescope_parameter_data(telescope_model_lst):

    read_parameters = ReadParameters(telescope_model=telescope_model_lst)

    result = read_parameters.get_telescope_parameter_data()

    # Assert the result contains the expected data
    if result[1] == "focal_length":
        assert result[0] == "Structure"
        assert result[2] == (2800.0 * u.cm)
        assert result[3] == "Nominal overall focal length of the entire telescope."
