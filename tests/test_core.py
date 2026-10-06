import numpy as np, pandas as pd
import mlflow
from xgboost import XGBRegressor
from src.metrics import rmse, nasa_score, maintenance_cost
from src.conformal import conformal_quantile, coverage
from src.monitor import psi, drift_report
from src.features import build_features, SENSORS
from src.train import log_serving_model


def test_metrics_asymmetry():
    y = np.array([50.0])
    assert nasa_score(y, y + 10) > nasa_score(y, y - 10)       # late is worse than early
    assert maintenance_cost(y, y + 10) > maintenance_cost(y, y - 10)
    assert rmse(y, y) == 0


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000); p = y + rng.normal(0, 1, 5000)
    q = conformal_quantile(y[:2500], p[:2500], 0.1)
    cov = coverage(y[2500:], p[2500:] - q, p[2500:] + q)
    assert 0.87 < cov < 0.93


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 3000)
    assert psi(a, rng.normal(0, 1, 3000)) < 0.1
    assert psi(a, rng.normal(2, 1, 3000)) > 0.2


def test_features_do_not_leak_across_engines():
    rng = np.random.default_rng(1)
    rows = []
    for u in (1, 2):
        for c in range(1, 21):
            rows.append({"unit": u, "cycle": c, **{s: u * 100.0 + rng.normal() for s in SENSORS}})
    f = build_features(pd.DataFrame(rows))
    first_of_unit2 = f[(f.unit == 2) & (f.cycle == 1)].iloc[0]
    assert abs(first_of_unit2[SENSORS[0] + "_mean"] - 200) < 5   # not contaminated by engine 1


def test_drift_report_fires():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"a": rng.normal(3, 1, 2000), "b": rng.normal(3, 1, 2000)})
    assert drift_report(ref, cur, ["a", "b"])["retrain"]


def test_log_serving_model_serializes_xgboost(tmp_path):
    model = XGBRegressor(n_estimators=2, max_depth=2, random_state=0).fit(
        np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0])
    )
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "artifacts")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("xgboost-model-logging", artifact_location=artifact_location)
    mlflow.set_experiment(experiment_id)

    with mlflow.start_run():
        log_serving_model(model, "model")
        run_id = mlflow.active_run().info.run_id

    logged_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    original = model.predict(np.array([[1.5]]))[0]
    restored = logged_model.predict(np.array([[1.5]]))[0]
    assert np.isclose(original, restored)
