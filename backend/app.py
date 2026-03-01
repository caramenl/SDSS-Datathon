from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Model" / "airfare_gbr.joblib"
COLS_PATH  = ROOT / "Model" / "feature_columns.joblib"

app = FastAPI(title="Airfare Prediction API", version="1.1")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(COLS_PATH)

class PredictRequest(BaseModel):
    nsmiles: float = Field(..., gt=0)
    passengers: float = Field(..., ge=0)
    large_ms: float = Field(..., ge=0, le=1)
    lf_ms: float = Field(..., ge=0, le=1)
    hub_intensity: int = Field(..., ge=0, le=2)
    Year: int = Field(..., ge=2021, le=2030)

class BatchPredictRequest(BaseModel):
    rows: List[PredictRequest]

@app.get("/")
def home():
    return {"message": "API running. Visit /docs", "endpoints": ["/health", "/predict", "/predict_batch"]}

@app.get("/health")
def health():
    return {"status": "ok"}

def build_features(req: PredictRequest) -> pd.DataFrame:
    row = pd.DataFrame([{
        "log_distance": float(np.log(req.nsmiles)),
        "log_passengers": float(np.log(req.passengers + 1.0)),
        "large_ms": float(req.large_ms),
        "lf_ms": float(req.lf_ms),
        "hub_intensity": int(req.hub_intensity),
        "Year": int(req.Year),
    }])

    row = pd.get_dummies(row, columns=["Year"], drop_first=True)
    row = row.reindex(columns=feature_columns, fill_value=0)
    return row

@app.post("/predict")
def predict(req: PredictRequest):
    X_row = build_features(req)
    log_fare = float(model.predict(X_row)[0])
    fare = float(np.exp(log_fare))
    return {"predicted_log_fare": log_fare, "predicted_fare": fare}

@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    X = pd.concat([build_features(r) for r in req.rows], ignore_index=True)
    log_fares = model.predict(X).astype(float)
    fares = np.exp(log_fares).astype(float)
    return {
        "predicted_log_fares": log_fares.tolist(),
        "predicted_fares": fares.tolist(),
        "n": len(fares),
    }