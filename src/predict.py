"""Inference helper shared by the API and the dashboard (no web-framework dependency)."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .features import build_features, SENSORS
from .data import RUL_CAP

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_rul(readings):
    """readings: list of dicts (one per cycle, oldest first) with `cycle` and the sensor columns."""
    df = pd.DataFrame(readings)
    missing = [c for c in ["cycle", *SENSORS] if c not in df.columns]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    df["unit"] = 1
    feats = build_features(df).tail(1)
    b = bundle()
    p = float(np.clip(b["model"].predict(b["scaler"].transform(feats[b["cols"]]))[0], 0, RUL_CAP))
    q = b["conformal_q"]
    return {"rul": p, "lower": max(0.0, p - q), "upper": min(float(RUL_CAP), p + q),
            "confidence": 1 - b["alpha"], "model": b["name"]}
