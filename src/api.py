"""FastAPI inference service.  Run: uvicorn src.api:app --port 8000"""
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .predict import predict_rul, bundle

app = FastAPI(title="Predictive Maintenance RUL API", version="1.0.0")


class PredictRequest(BaseModel):
    # One dict per operating cycle (oldest first): {"cycle": 1, "setting1": ..., "s2": ..., ...}
    readings: List[Dict[str, float]]


@app.get("/health")
def health():
    b = bundle()
    return {"status": "ok", "model": b["name"], "val_rmse": b["val_rmse"]}


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.readings:
        raise HTTPException(422, "readings must not be empty")
    try:
        return predict_rul(req.readings)
    except ValueError as e:
        raise HTTPException(422, str(e))
