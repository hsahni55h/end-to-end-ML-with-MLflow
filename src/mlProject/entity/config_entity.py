from dataclasses import dataclass  # Import the dataclass decorator for creating data classes
from pathlib import Path  # Import the Path class for handling filesystem paths

@dataclass(frozen=True)  # Define the class as a dataclass and make it immutable with frozen=True
class DataIngestionConfig:
    # Configuration class for data ingestion process
    root_dir: Path  # The root directory where data ingestion operations will take place
    source_URL: str  # The URL from which the data will be downloaded
    local_data_file: Path  # The local path where the downloaded data will be stored
    unzip_dir: Path  # The directory where the downloaded data will be unzipped
