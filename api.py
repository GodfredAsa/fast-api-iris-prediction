from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import numpy as np
import io

app = FastAPI()

# Load the trained model and selector
model = joblib.load('iris_model.pkl')
selector = joblib.load('selector.pkl')

# Example data format for /data endpoint
data_example = [
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "petal_ratio": 7.0
    },
    {
        "sepal_length": 6.2,
        "sepal_width": 2.8,
        "petal_length": 4.8,
        "petal_width": 1.8,
        "petal_ratio": 2.67
    }
]

@app.get("/data")
def get_data_example():
    """Returns a JSON example of the expected data format for predictions."""
    return JSONResponse(content={"expected_format": data_example})

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Accepts a CSV file and returns predictions for each row."""
    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    # Check for required columns
    required_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing required column: {col}")

    # Add petal_ratio if not present
    if "petal_ratio" not in df.columns:
        df["petal_ratio"] = df["petal_length"] / df["petal_width"]

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width", "petal_ratio"]
    X = df[features]
    # Apply the same feature selection as during training
    X_selected = selector.transform(X)
    preds = model.predict(X_selected)
    return {"predictions": preds.tolist()} 