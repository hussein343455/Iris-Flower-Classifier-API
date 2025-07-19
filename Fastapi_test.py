from typing import Optional

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from pathlib import Path

# Gets the absolute path to the directory containing the current file (fastapi.py)
BASE_DIR = Path(__file__).resolve().parent

# uvicorn main: app --reload

# Securely joins the base path with the rest of the path
MODEL_PATH = BASE_DIR / "Models" / "iris_classifier.joblib"
# Load the trained model when the app starts
iris_model = joblib.load(MODEL_PATH)
# You might also have a label encoder or class names
class_names = np.array(['setosa', 'versicolor', 'virginica'])

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    # In your main.py, after defining IrisFeatures and app


app = FastAPI()

@app.post("/predict/")
def create_prediction(iris: IrisFeatures):
    # 1. Convert incoming data into a 2D numpy array
    # The model expects a list of samples, so we create a list with one sample
    features = np.array([[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]])

    # 2. Make a prediction
    # model.predict() returns an array (e.g., [0]), so we get the first element
    prediction_index = iris_model.predict(features)[0]

    # 3. Get the class name from the index
    prediction_name = class_names[prediction_index]

    # 4. Return the prediction
    return {"prediction": prediction_name}
