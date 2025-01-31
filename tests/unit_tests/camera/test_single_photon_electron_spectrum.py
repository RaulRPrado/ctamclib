#!/usr/bin/python3

import copy
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
from astropy.table import Table

from simtools.camera.single_photon_electron_spectrum import SinglePhotonElectronSpectrum


@pytest.fixture
def spe_spectrum():
    args_dict = {
        "output_file": "output_file",
        "simtel_path": "/path/to/simtel",
        "step_size": 0.1,
        "max_amplitude": 1.0,
        "afterpulse_spectrum": None,
        "input_spectrum": "input_spectrum",
    }
    return SinglePhotonElectronSpectrum(args_dict)


@pytest.fixture
def spe_data():
    return "0.0,0.4694\n0.02,0.46378\n0.04,0.45267\n0.06,0.44172"


@patch("simtools.io_operations.io_handler.IOHandler")
@patch("simtools.camera.single_photon_electron_spectrum.MetadataCollector")
def test_init(mock_metadata_collector, mock_io_handler, spe_spectrum):

    mock_io_handler_instance = mock_io_handler.return_value
    mock_metadata_collector_instance = mock_metadata_collector.return_value

    spe_spectrum.io_handler = mock_io_handler_instance
    spe_spectrum.metadata = mock_metadata_collector_instance
    tmp_spe_spectrum = copy.deepcopy(spe_spectrum)

    assert tmp_spe_spectrum.args_dict["output_file"] == "output_file.ecsv"
    assert tmp_spe_spectrum.io_handler == mock_io_handler_instance
    assert tmp_spe_spectrum.data == ""
    assert tmp_spe_spectrum.metadata == mock_metadata_collector_instance


@patch(
    "simtools.camera.single_photon_electron_spectrum.SinglePhotonElectronSpectrum._derive_spectrum_norm_spe"
)
def test_derive_single_pe_spectrum(mock_derive_spectrum_norm_spe, spe_spectrum):

    spe_spectrum.args_dict["use_norm_spe"] = True
    spe_spectrum.derive_single_pe_spectrum()
    mock_derive_spectrum_norm_spe.assert_called_once()

    spe_spectrum.args_dict["use_norm_spe"] = False
    with pytest.raises(
        NotImplementedError,
        match="Derivation of single photon electron spectrum using a simtool is not implemented.",
    ):
        spe_spectrum.derive_single_pe_spectrum()


@patch("simtools.camera.single_photon_electron_spectrum.io_handler.IOHandler.get_output_directory")
@patch("simtools.camera.single_photon_electron_spectrum.writer.ModelDataWriter.dump")
@patch("builtins.open", new_callable=MagicMock)
def test_write_single_pe_spectrum(mock_open, mock_dump, mock_get_output_directory, spe_spectrum):
    mock_get_output_directory.return_value = Path("/mock/output/directory")
    mock_open.return_value.__enter__.return_value = MagicMock()

    tmp_spe_spectrum = copy.deepcopy(spe_spectrum)

    tmp_spe_spectrum.data = """
# comment
0.0\t0.4694\t0.4694
0.02\t0.46378\t0.46378
0.04\t0.45267\t0.45267
0.06\t0.44172\t0.44172
"""
    tmp_spe_spectrum.write_single_pe_spectrum()

    mock_open.assert_called_once_with(
        Path("/mock/output/directory/output_file.dat"), "w", encoding="utf-8"
    )
    mock_dump.assert_called_once()


@patch("simtools.camera.single_photon_electron_spectrum.subprocess.run")
@patch(
    "simtools.camera.single_photon_electron_spectrum.SinglePhotonElectronSpectrum._get_input_data"
)
def test_derive_spectrum_norm_spe(mock_get_input_data, mock_subprocess_run, spe_spectrum, spe_data):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmpfile:
        tmpfile.write(spe_data)
    # first call to _get_input_data returns tmpfile, second call None
    mock_get_input_data.side_effect = [tmpfile, None]
    mock_subprocess_run.return_value.stdout = spe_data
    mock_subprocess_run.return_value.returncode = 0

    return_code = spe_spectrum._derive_spectrum_norm_spe()

    assert mock_get_input_data.call_count == 2
    mock_subprocess_run.assert_called_once_with(
        ["/path/to/simtel/sim_telarray/bin/norm_spe", "-r", "0.1,1.0", ANY],
        capture_output=True,
        text=True,
        check=True,
    )
    assert return_code == 0
    assert spe_spectrum.data == spe_data

    tmp_spe_spectrum = copy.deepcopy(spe_spectrum)
    tmp_spe_spectrum.args_dict["afterpulse_spectrum"] = "afterpulse_spectrum"
    mock_get_input_data.side_effect = [tmpfile, tmpfile]
    tmp_spe_spectrum._derive_spectrum_norm_spe()
    mock_subprocess_run.assert_called_with(
        [
            "/path/to/simtel/sim_telarray/bin/norm_spe",
            "-a",
            ANY,
            "-r",
            "0.1,1.0",
            ANY,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # test error handling
    spe_spectrum = copy.deepcopy(spe_spectrum)
    mock_get_input_data.side_effect = [tmpfile, None]
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="norm_spe", output="Error", stderr="Error message"
    )
    with pytest.raises(subprocess.CalledProcessError):
        spe_spectrum._derive_spectrum_norm_spe()


@patch("builtins.open", new_callable=MagicMock)
def test_get_input_data(mock_open, spe_spectrum, spe_data):
    assert spe_spectrum._get_input_data(None, spe_spectrum.prompt_column) is None

    mock_open.return_value.__enter__.return_value.read.return_value = spe_data.replace(" ", ",")
    input_data = spe_spectrum._get_input_data("input_spectrum", spe_spectrum.prompt_column)
    mock_open.assert_called_once_with(Path("input_spectrum"), encoding="utf-8")
    assert input_data is not None
    with open(input_data.name, encoding="utf-8") as f:
        assert f.read() == spe_data

    input_data = spe_spectrum._get_input_data("input_spectrum", spe_spectrum.afterpulse_column)
    assert input_data is not None
    with open(input_data.name, encoding="utf-8") as f:
        assert f.read() == spe_data.replace(" ", ",")

    with patch("astropy.table.Table.read") as mock_table_read:
        mock_table = Table()
        mock_table["amplitude"] = [0.0, 0.02, 0.04, 0.06]
        mock_table["frequency (prompt)"] = [0.4694, 0.46378, 0.45267, 0.44172]
        mock_table_read.return_value = mock_table

        ecsv_data = spe_spectrum._get_input_data("input_spectrum.ecsv", spe_spectrum.prompt_column)
        assert ecsv_data is not None
        with open(ecsv_data.name, encoding="utf-8") as f:
            table_data = f.read()
            assert table_data.splitlines()[0] == "0.0,0.4694"
