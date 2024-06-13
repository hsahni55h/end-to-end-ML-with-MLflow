import os
from mlProject import logger
import pandas as pd
from mlProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        """
        Initialize the DataValidation class with a configuration.
        
        Args:
            config (DataValidationConfig): Configuration for data validation.
        """
        self.config = config  # Store the configuration

    def validate_all_columns(self) -> bool:
        """
        Validate that all columns in the dataset match the expected schema and their data types are correct.
        
        Returns:
            bool: True if all columns and their data types are valid, False otherwise.
        """
        try:
            validation_status = True  # Initialize validation status
            
            # Read the CSV file from the unzipped data directory
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)  # Get the list of columns in the data

            # Get the list of expected columns and their data types from the schema
            all_schema = self.config.all_schema

            # Validate each column in the data
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False  # Set validation status to False if a column is not in the schema
                    logger.error(f"Column {col} is not in the schema")
                else:
                    # Check the data type of the column
                    expected_dtype = all_schema[col]
                    actual_dtype = str(data[col].dtype)
                    if actual_dtype != expected_dtype:
                        validation_status = False  # Set validation status to False if data type does not match
                        logger.error(f"Data type for column {col} does not match. Expected: {expected_dtype}, Found: {actual_dtype}")

            # Write the final validation status to the status file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status  # Return the final validation status
        
        except Exception as e:
            raise e  # Raise the exception if any error occurs
