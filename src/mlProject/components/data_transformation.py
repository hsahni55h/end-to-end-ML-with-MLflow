import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd 
from mlProject.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Initialize the DataTransformation class with a configuration.
        
        Args:
            config (DataTransformationConfig): Configuration for data transformation.
        """
        self.config = config  # Store the configuration

    def train_test_spliting(self):
        """
        Split the data into training and test sets and save them to files.
        """
        try:
            # Load the data from the specified data path
            data = pd.read_csv(self.config.data_path)

            # Split the data into training and test sets with a 75-25 split
            train, test = train_test_split(data, test_size=0.25, random_state=42)

            # Save the training data to a CSV file in the root directory
            train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            
            # Save the test data to a CSV file in the root directory
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

            # Log the success of the data split
            logger.info("Split data into training and test sets")
            logger.info(f"Training data shape: {train.shape}")
            logger.info(f"Test data shape: {test.shape}")

            # Print the shapes of the training and test sets
            print(f"Training data shape: {train.shape}")
            print(f"Test data shape: {test.shape}")

        except Exception as e:
            logger.error(f"Error during train-test split: {e}")
            raise e
