import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize the ModelTrainer class with a configuration.
        
        Args:
            config (ModelTrainerConfig): Configuration for model training.
        """
        self.config = config  # Store the configuration

    def train(self):
        """
        Train the ElasticNet model using the training data and save the model.
        """
        # Load the training and testing data from the specified paths
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Split the data into features (X) and target (y) for training and testing
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        # Initialize the ElasticNet model with the specified alpha and l1_ratio
        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        
        # Fit the model to the training data
        lr.fit(train_x, train_y)

        # Save the trained model to the specified path
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
