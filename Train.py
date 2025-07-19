import os

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'],
                                                    test_size=0.2, random_state=42)

# Using a simple, robust model as a starting point
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# --- Model Saving ---
# Ensure the 'Models' directory exists
os.makedirs("Models", exist_ok=True)

# --- Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

joblib.dump(model, "Models/iris_classifier.joblib")
print("Model saved successfully to 'Models/iris_classifier.joblib'")

