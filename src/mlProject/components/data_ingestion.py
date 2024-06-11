import os
import urllib.request as request
import zipfile
from pathlib import Path
from mlProject import logger
from mlProject.utils.common import get_size
from mlProject.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the DataIngestion class with a configuration.
        
        Args:
            config (DataIngestionConfig): Configuration for data ingestion.
        """
        self.config = config  # Store the configuration

    def download_file(self):
        """
        Downloads the file from the source URL if it does not already exist locally.
        """
        if not os.path.exists(self.config.local_data_file):
            # If the file does not exist, download it from the source URL
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            # If the file already exists, log its size
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        Extracts the zip file into the unzip directory.
        """
        unzip_path = self.config.unzip_dir  # Get the directory where the zip file will be extracted
        os.makedirs(unzip_path, exist_ok=True)  # Ensure the unzip directory exists
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)  # Extract all contents of the zip file into the unzip directory
