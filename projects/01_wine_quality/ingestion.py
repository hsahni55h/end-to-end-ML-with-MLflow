"""wine_quality Ingestor: downloads and loads the UCI winequality-red dataset.

Subclasses `core.ingestion.base.Ingestor[pd.DataFrame]` — see that module's
docstring for how this fits the generic Ingestor contract.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from core.ingestion.base import Ingestor


class WineQualityIngestor(Ingestor[pd.DataFrame]):
    """Downloads the winequality-red zip and loads the extracted CSV."""

    def __init__(
        self,
        source_url: str,
        local_data_file: Path,
        unzip_dir: Path,
        csv_filename: str = "winequality-red.csv",
    ) -> None:
        self.source_url = source_url
        self.local_data_file = local_data_file
        self.unzip_dir = unzip_dir
        self.csv_filename = csv_filename

    def fetch(self) -> None:
        """Download + extract the dataset, unless the CSV is already there.

        Checking for the final extracted CSV (rather than just the zip)
        makes this idempotent end-to-end: a second call does no network
        I/O and no re-extraction, not just no re-download.
        """
        csv_path = self.unzip_dir / self.csv_filename
        if csv_path.exists():
            return

        self.local_data_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.local_data_file.exists():
            urllib.request.urlretrieve(self.source_url, self.local_data_file)

        self.unzip_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.local_data_file, "r") as zip_ref:
            zip_ref.extractall(self.unzip_dir)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.unzip_dir / self.csv_filename)
