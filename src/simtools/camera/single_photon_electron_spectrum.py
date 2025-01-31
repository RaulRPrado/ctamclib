"""Single photon electron spectral analysis."""

import logging
import re
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from astropy.table import Table

import simtools.data_model.model_data_writer as writer
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io_operations import io_handler


class SinglePhotonElectronSpectrum:
    """
    Single photon electron spectral analysis.

    Parameters
    ----------
    args_dict: dict
        Dictionary with input arguments.
    """

    prompt_column = "frequency (prompt)"
    afterpulse_column = "frequency (afterpulsing)"

    def __init__(self, args_dict):
        """Initialize SinglePhotonElectronSpectrum class."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initialize SinglePhotonElectronSpectrum class.")

        self.args_dict = args_dict
        # default output is of ecsv format
        self.args_dict["output_file"] = str(
            Path(self.args_dict["output_file"]).with_suffix(".ecsv")
        )
        self.io_handler = io_handler.IOHandler()
        self.data = ""  # Single photon electron spectrum data (as string)
        self.metadata = MetadataCollector(args_dict=self.args_dict)

    def derive_single_pe_spectrum(self):
        """Derive single photon electron spectrum."""
        if self.args_dict.get("use_norm_spe"):
            return self._derive_spectrum_norm_spe()

        raise NotImplementedError(
            "Derivation of single photon electron spectrum using a simtool is not implemented."
        )

    def write_single_pe_spectrum(self):
        """
        Write single photon electron spectrum plus metadata to disk.

        Includes writing in simtel and simtools (ecsv) formats.

        """
        simtel_file = self.io_handler.get_output_directory() / Path(
            self.args_dict["output_file"]
        ).with_suffix(".dat")
        self._logger.debug(f"norm_spe output file: {simtel_file}")
        with open(simtel_file, "w", encoding="utf-8") as simtel:
            simtel.write(self.data)

        cleaned_data = re.sub(r"%%%.+", "", self.data)  # remove norm_spe row metadata
        table = Table.read(
            BytesIO(cleaned_data.encode("utf-8")),
            format="ascii.no_header",
            comment="#",
            delimiter="\t",
        )
        table.rename_columns(
            ["col1", "col2", "col3"],
            ["amplitude", self.prompt_column, self.afterpulse_column],
        )

        writer.ModelDataWriter.dump(
            args_dict=self.args_dict,
            metadata=self.metadata.top_level_meta,
            product_data=table,
            validate_schema_file=None,
        )

    def _derive_spectrum_norm_spe(self):
        """
        Derive single photon electron spectrum using sim_telarray tool 'norm_spe'.

        Returns
        -------
        int
            Return code of the executed command

        Raises
        ------
        subprocess.CalledProcessError
            If the command execution fails.
        """
        tmp_input_file = self._get_input_data(
            input_file=self.args_dict["input_spectrum"],
            frequency_column=self.prompt_column,
        )
        tmp_ap_file = self._get_input_data(
            input_file=self.args_dict.get("afterpulse_spectrum"),
            frequency_column=self.afterpulse_column,
        )

        command = [
            f"{self.args_dict['simtel_path']}/sim_telarray/bin/norm_spe",
            "-r",
            f"{self.args_dict['step_size']},{self.args_dict['max_amplitude']}",
            tmp_input_file.name,
        ]
        if tmp_ap_file:
            command.insert(1, "-a")
            command.insert(2, f"{tmp_ap_file.name}")

        self._logger.debug(f"Running norm_spe command: {' '.join(command)}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            self._logger.error(f"Error running norm_spe: {exc}")
            self._logger.error(f"stderr: {exc.stderr}")
            raise exc
        finally:
            for tmp_file in [tmp_input_file, tmp_ap_file]:
                try:
                    Path(tmp_file.name).unlink()
                except (AttributeError, FileNotFoundError):
                    pass

        self.data = result.stdout
        return result.returncode

    def _get_input_data(self, input_file, frequency_column):
        """
        Return input data for norm_spe command.

        Input data need to be space separated values of the amplitude spectrum.
        """
        input_data = ""
        if not input_file:
            return None
        input_file = Path(input_file)

        if input_file.suffix == ".ecsv":
            table = Table.read(input_file)
            input_data = "\n".join(f"{row['amplitude']} {row[frequency_column]}" for row in table)
        else:
            with open(input_file, encoding="utf-8") as f:
                input_data = (
                    f.read().replace(",", " ")
                    if frequency_column == self.prompt_column
                    else f.read()
                )

        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmpfile:
            tmpfile.write(input_data)
        return tmpfile
