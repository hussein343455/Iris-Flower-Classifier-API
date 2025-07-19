# tests/test_main.py
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
from Fastapi_test import app


client = TestClient(app)

# -- Tests for the /predict/ endpoint --
import pytest

@pytest.mark.parametrize("payload, mock_prediction, expected_name", [
    (
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        np.array([0]), # Mock model predicts index 0
        "setosa"
    ),
    (
        {"sepal_length": 6.0, "sepal_width": 2.2, "petal_length": 4.0, "petal_width": 1.0},
        np.array([1]), # Mock model predicts index 1
        "versicolor"
    ),
    (
        {"sepal_length": 7.3, "sepal_width": 2.9, "petal_length": 6.3, "petal_width": 1.8},
        np.array([2]), # Mock model predicts index 2
        "virginica"
    )
])
# The @patch decorator replaces the model's predict method with a mock during the test
@patch('Fastapi_test.iris_model.predict')
def test_predict_success(mock_predict, payload, mock_prediction, expected_name):
    """
    Tests the /predict endpoint for successful predictions.
    """
    # Configure the mock to return the desired prediction index
    mock_predict.return_value = mock_prediction

    # Make a request to the API
    response = client.post("/predict/", json=payload)

    # Assert the response is what we expect
    assert response.status_code == 200
    assert response.json() == {"prediction": expected_name}

    # You can also assert that your mock was called correctly
    mock_predict.assert_called_once()


def test_predict_validation_error():
    """
    Tests that a 422 Unprocessable Entity error is returned for invalid data.
    """
    # This payload is missing 'petal_width'
    invalid_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
    }
    response = client.post("/predict/", json=invalid_payload)
    assert response.status_code == 422

