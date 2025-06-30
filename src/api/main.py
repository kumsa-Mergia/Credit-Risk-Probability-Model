from fastapi import FastAPI
from src.api.pydantic_models import CustomerInput, RiskPrediction
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Credit Risk Prediction API")

# Load model from MLflow registry
model = mlflow.pyfunc.load_model("models:/RandomForest/Production")


@app.get("/")
def root():
    return {"message": "Credit Risk API is running."}


@app.post("/predict", response_model=RiskPrediction)
def predict_risk(customer: CustomerInput):
    df = pd.DataFrame([customer.dict()])
    prediction = model.predict(df)[0]
    return {"risk_probability": float(prediction)}
