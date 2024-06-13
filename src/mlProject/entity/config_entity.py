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


@dataclass(frozen=True)  # Define the class as a dataclass and make it immutable with frozen=True
class DataTransformationConfig:
    # Configuration class for data transformation.

    root_dir: Path # root_dir (Path): The root directory where data transformation operations will take place.
    data_path: Path # data_path (Path): The path to the data that will be transformed.

@dataclass(frozen=True)  # Define the class as a dataclass and make it immutable with frozen=True
class ModelTrainerConfig:
    """
    Configuration class for model training. 
    """
    root_dir: Path # root_dir (Path): The root directory where model training operations will take place.
    train_data_path: Path # train_data_path (Path): The path to the training data.
    test_data_path: Path # test_data_path (Path): The path to the testing data.
    model_name: str # model_name (str): The name of the model to be trained.
    alpha: float # alpha (float): The regularization strength parameter for the model.
    l1_ratio: float # l1_ratio (float): The ElasticNet mixing parameter.
    target_column: str # target_column (str): The name of the target column in the dataset.
