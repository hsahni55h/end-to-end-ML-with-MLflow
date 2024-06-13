from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig)



class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH, schema_filepath=SCHEMA_FILE_PATH):
        """
        Initialize ConfigurationManager with paths to the configuration files.
        
        Args:
            config_filepath (Path): Path to the configuration YAML file.
            params_filepath (Path): Path to the parameters YAML file.
            schema_filepath (Path): Path to the schema YAML file.
        """
        self.config = read_yaml(config_filepath)  # Read the configuration file
        self.params = read_yaml(params_filepath)  # Read the parameters file
        self.schema = read_yaml(schema_filepath)  # Read the schema file

        create_directories([self.config.artifacts_root])  # Create the root directory for artifacts
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get the DataIngestionConfig object from the configuration.
        
        Returns:
            DataIngestionConfig: Configuration for data ingestion.
        """
        config = self.config.data_ingestion  # Get the data ingestion configuration section

        create_directories([config.root_dir])  # Ensure the root directory for data ingestion exists

        # Create and return a DataIngestionConfig object with the configuration details
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config


    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Get the DataValidationConfig object from the configuration.
        
        Returns:
            DataValidationConfig: Configuration for data validation.
        """
        config = self.config.data_validation  # Get the data validation configuration section
        schema = self.schema.COLUMNS  # Get the schema columns

        create_directories([config.root_dir])  # Ensure the root directory for data validation exists

        # Create and return a DataValidationConfig object with the configuration details
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
