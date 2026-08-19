from pathlib import Path  # Import Path for filesystem paths

import joblib  # Import joblib to load the trained model
import numpy as np  # Import numpy for numerical operations
from flask import (  # Import Flask for creating the web app
    Flask,
    render_template,
    request,
)

app = Flask(__name__)  # Initialize a Flask app


class PredictionPipeline:
    """Loads the trained wine_quality model and serves predictions.

    Inlined here (rather than imported from the retired legacy
    `src/mlProject` package) since this is the only piece of that package
    `app.py` still needed. `app.py` itself remains as the reference
    implementation for the FastAPI migration in S1.3.
    """

    def __init__(self):
        # Load the model from the specified path using joblib
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))

    def predict(self, data):
        # Make predictions using the loaded model
        prediction = self.model.predict(data)
        return prediction


@app.route("/", methods=["GET"])  # Route to display the home page
def homePage():
    return render_template("index.html")  # Render the home page template


@app.route("/train", methods=["GET"])  # Route to train the pipeline
def training():
    # main.py (the legacy src/mlProject training entrypoint) was retired in
    # S1.2. The new training entrypoint is
    # `uv run python -m projects.01_wine_quality.pipeline`, but it saves its
    # model under projects/01_wine_quality/artifacts/, not the root
    # artifacts/ directory `/predict` below reads from — wiring this route
    # to it directly would silently retrain a model `/predict` never sees.
    # Left as a stub pending the S1.3 FastAPI migration, which replaces
    # this training/serving split entirely.
    return (
        "Training via this route is retired. Run "
        "`uv run python -m projects.01_wine_quality.pipeline` directly instead."
    )


@app.route(
    "/predict", methods=["POST", "GET"]
)  # Route to show the predictions in a web UI
def index():
    if request.method == "POST":
        try:
            # Reading the inputs given by the user
            fixed_acidity = float(request.form["fixed_acidity"])
            volatile_acidity = float(request.form["volatile_acidity"])
            citric_acid = float(request.form["citric_acid"])
            residual_sugar = float(request.form["residual_sugar"])
            chlorides = float(request.form["chlorides"])
            free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
            total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
            density = float(request.form["density"])
            pH = float(request.form["pH"])
            sulphates = float(request.form["sulphates"])
            alcohol = float(request.form["alcohol"])

            # Creating a numpy array from the input values
            data = [
                fixed_acidity,
                volatile_acidity,
                citric_acid,
                residual_sugar,
                chlorides,
                free_sulfur_dioxide,
                total_sulfur_dioxide,
                density,
                pH,
                sulphates,
                alcohol,
            ]
            data = np.array(data).reshape(1, 11)

            # Initialize the prediction pipeline and make predictions
            obj = PredictionPipeline()
            predict = obj.predict(data)

            # Render the results page with the prediction
            return render_template("results.html", prediction=str(predict))

        except Exception as e:
            print("The Exception message is: ", e)
            return "Something is wrong"

    else:
        return render_template("index.html")


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port = 8080, debug=True)
    app.run(host="0.0.0.0", port=8080)
