# Predictive Maintenance & Remaining Useful Life (RUL) Platform

End-to-end ML system that predicts how many operating cycles an industrial turbofan engine has left,
from multivariate sensor time series (NASA C-MAPSS, FD001). It covers the full lifecycle: feature
engineering, model comparison, uncertainty quantification, cost-aware evaluation, experiment tracking,
serving, monitoring and documentation.

## Pipeline
| Stage | What it does |
|---|---|
| Features (`src/features.py`) | Per-engine rolling mean/std (10 cycles) and deviation from early-life level. No leakage across engines. |
| Models (`src/train.py`) | Ridge baseline vs XGBoost vs optional LSTM (`--lstm`). Engines are split train/calibration/validation. |
| Evaluation (`src/metrics.py`) | RMSE, official NASA score, and an asymmetric maintenance-cost metric (late prediction = 5x cost of early). |
| Uncertainty (`src/conformal.py`) | Split conformal prediction: 90% intervals, empirical coverage reported on the official test set. |
| Tracking | MLflow logs params, metrics and the serving model for every run. |
| Serving (`src/api.py`, `app/dashboard.py`) | FastAPI `/predict` + Streamlit dashboard, Dockerised. |
| Monitoring (`src/monitor.py`) | KS-test + PSI per feature, performance-degradation check, retraining triggers. |
| Docs (`docs/`) | Model card, risk register, change log. |

## Quick start
1. Download C-MAPSS (NASA Prognostics Data Repository, "Turbofan Engine Degradation Simulation", also mirrored on Kaggle)
   and copy `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` into `data/raw/`.
2. `pip install -r requirements.txt`
3. `python -m src.train --lstm` -> prints the headline numbers, writes `results.json` and `models/model.joblib`
4. `mlflow ui` to browse runs
5. `uvicorn src.api:app --port 8000` (docs at `/docs`) or `streamlit run app/dashboard.py`
6. `python -m scripts.simulate_drift` to watch the retraining trigger fire
7. `docker build -t rul-api . && docker run -p 8000:8000 rul-api`

`scripts/make_synthetic.py` generates fake data in the same format to smoke-test the code. Never report numbers from it.

## Results
Fill in from `results.json` after your real run:

| Model | Val RMSE | Test RMSE | NASA score | Maintenance cost |
|---|---|---|---|---|
| Ridge baseline | | | | |
| XGBoost | | | | |
| LSTM | | | | |

90% conformal interval empirical test coverage: ___%

## Design decisions
- Target RUL is capped at 125 cycles (standard piece-wise linear target): early life carries no degradation signal.
- The serving model is selected on validation RMSE, never on the test set.
- The LSTM is a benchmark only; the API serves the best tabular model to keep the container light.
- Cost weights (1 early / 5 late) are assumptions. Change them in `src/metrics.py` for a real client.
