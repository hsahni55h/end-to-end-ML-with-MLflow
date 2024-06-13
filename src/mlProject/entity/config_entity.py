from dataclasses import dataclass  # Import the dataclass decorator for creating data classes
from pathlib import Path  # Import the Path class for handling filesystem paths

@dataclass(frozen=True)  # Define the class as a dataclass and make it immutable with frozen=True
class DataIngestionConfig:
    # Configuration class for data ingestion process
    root_dir: Path  # The root directory where data ingestion operations will take place
    source_URL: str  # The URL from which the data will be downloaded
    local_data_file: Path  # The local path where the downloaded data will be stored
    unzip_dir: Path  # The directory where the downloaded data will be unzipped


@dataclass(frozen=True)  # Define the class as a dataclass and make it immutable with frozen=True
class DataValidationConfig:
    # Configuration class for data validation.
    root_dir: Path # root_dir (Path): The root directory where data validation operations will take place.
    STATUS_FILE: str # STATUS_FILE (str): The name of the status file that will store the status of the data validation process.
    unzip_data_dir: Path # unzip_data_dir (Path): The directory where the unzipped data is stored.
    all_schema: dict # all_schema (dict): A dictionary containing all the schema definitions for the data to be validated.