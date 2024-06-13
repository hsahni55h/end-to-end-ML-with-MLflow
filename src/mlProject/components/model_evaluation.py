import os
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
import joblib  # Import joblib for loading the model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Import metrics for evaluation
import mlflow  # Import mlflow for tracking and logging
import mlflow.sklearn
from urllib.parse import urlparse  # Import urlparse for parsing URIs
from pathlib import Path  # Import Path class for handling filesystem paths
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize the ModelEvaluation class with a configuration.
        
        Args:
            config (ModelEvaluationConfig): Configuration for model evaluation.
        """
        self.config = config  # Store the configuration

    def eval_metrics(self, actual, pred):
        """
        Evaluate the model performance using RMSE, MAE, and R2 score.
        
        Args:
            actual (array-like): Actual target values.
            pred (array-like): Predicted target values.
        
        Returns:
            tuple: RMSE, MAE, and R2 score.
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        """
        Log the model evaluation metrics and model into MLflow.
        """
        # Load the test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Split the test data into features and target
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Set the MLflow tracking URI
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Start an MLflow run
        with mlflow.start_run():
            # Predict the target values using the loaded model
            predicted_qualities = model.predict(test_x)

            # Evaluate the model performance
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Save metrics locally as JSON
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log parameters and metrics to MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Log the model to MLflow
            if tracking_url_type_store != "file":
                # Register the model if not using file store
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")
