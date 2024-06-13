class PredictionPipeline:
    def __init__(self):
        """
        Initialize the PredictionPipeline class.
        Loads the trained model from the specified path.
        """
        # Load the model from the specified path using joblib
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    def predict(self, data):
        """
        Make predictions on the provided data using the loaded model.

        Args:
            data (array-like or DataFrame): The input data for making predictions.

        Returns:
            array: The predicted values.
        """
        # Make predictions using the loaded model
        prediction = self.model.predict(data)
        return prediction
