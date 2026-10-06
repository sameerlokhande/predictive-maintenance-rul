"""Train + compare baseline / gradient boosting / LSTM on C-MAPSS FD001, track with MLflow,
add conformal intervals, save the serving model and print the numbers for the README / CV.

Usage:  python -m src.train [--lstm] [--seed 42]
"""
import argparse, json, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import load_train, load_test, RUL_CAP
from .features import build_features, feature_columns, SENSORS
from .metrics import rmse, nasa_score, maintenance_cost, pct_reduction
from .conformal import conformal_quantile, coverage

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ALPHA = 0.1
WIN = 30

try:
    import mlflow
except ImportError:  # keeps the pipeline runnable without MLflow installed
    mlflow = None


def run_ctx(name):
    if not mlflow:
        return contextlib.nullcontext()
    if mlflow.active_run():
        mlflow.end_run()  # never leave a stray run open
    return mlflow.start_run(run_name=name)


def log(params=None, metrics=None):
    if mlflow and mlflow.active_run():
        if params: mlflow.log_params(params)
        if metrics: mlflow.log_metrics(metrics)


def log_serving_model(model, artifact_path):
    """Log a sklearn model while explicitly trusting its known XGBoost classes."""
    if not mlflow:
        raise RuntimeError("MLflow is required to log the serving model")
    mlflow.sklearn.log_model(
        model,
        artifact_path,
        skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"],
    )


def cut_idx(feats, units, per_engine, rng, min_cycle=30):
    """Random 'cut points' per engine: mimics predicting at an arbitrary point in an engine's life."""
    out = []
    for u in units:
        rows = feats.index[(feats["unit"] == u) & (feats["cycle"] >= min_cycle)]
        out.extend(rng.choice(rows, size=min(per_engine, len(rows)), replace=False))
    return np.array(out)


def last_idx(feats):
    return feats.groupby("unit")["cycle"].idxmax().to_numpy()


def windows(feats, idx, sens_scaler):
    """(n, WIN, n_sensors) sequences ending at each row in idx; left-padded with the first reading."""
    X = np.zeros((len(idx), WIN, len(SENSORS)), dtype=np.float32)
    by_unit = {u: g for u, g in feats.groupby("unit")}
    for k, i in enumerate(idx):
        u, c = feats.at[i, "unit"], feats.at[i, "cycle"]
        g = by_unit[u]
        seq = g.loc[g["cycle"] <= c, SENSORS].tail(WIN).to_numpy()
        if len(seq) < WIN:
            seq = np.vstack([np.repeat(seq[:1], WIN - len(seq), axis=0), seq])
        X[k] = sens_scaler.transform(pd.DataFrame(seq, columns=SENSORS))
    return X


def train_lstm(feats, tr_idx, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    sens_scaler = StandardScaler().fit(feats.loc[tr_idx, SENSORS])
    Xtr = torch.tensor(windows(feats, tr_idx, sens_scaler))
    ytr = torch.tensor(feats.loc[tr_idx, "rul"].to_numpy(dtype=np.float32) / RUL_CAP)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(len(SENSORS), 64, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(30):
        net.train()
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(perm), 256):
            j = perm[b:b + 256]
            opt.zero_grad()
            loss = loss_fn(net(Xtr[j]), ytr[j])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

    def predict(f, idx):
        net.eval()
        with torch.no_grad():
            p = net(torch.tensor(windows(f, idx, sens_scaler))).numpy() * RUL_CAP
        return np.clip(p, 0, RUL_CAP)

    return predict


def main(use_lstm=False, seed=42):
    rng = np.random.default_rng(seed)
    MODELS.mkdir(exist_ok=True)

    raw = load_train().sort_values(["unit", "cycle"]).reset_index(drop=True)
    feats = build_features(raw)
    feats["rul"] = raw["rul"].to_numpy()
    cols = [c for c in feature_columns(feats) if c != "rul"]

    # Split BY ENGINE so no engine appears in more than one split.
    units = rng.permutation(feats["unit"].unique())
    n = len(units)
    tr_u, cal_u, val_u = units[: int(.7 * n)], units[int(.7 * n): int(.85 * n)], units[int(.85 * n):]
    tr_idx = feats.index[feats["unit"].isin(tr_u)].to_numpy()
    cal_idx = cut_idx(feats, cal_u, 5, rng)
    val_idx = cut_idx(feats, val_u, 5, rng)

    test_raw, test_rul = load_test()
    test_feats = build_features(test_raw)
    te_idx = last_idx(test_feats)
    y_te = test_rul.loc[test_feats.loc[te_idx, "unit"]].to_numpy()

    scaler = StandardScaler().fit(feats.loc[tr_idx, cols])
    Xs = lambda f, idx: scaler.transform(f.loc[idx, cols])
    ytr = feats.loc[tr_idx, "rul"]

    models = {}
    ridge = Ridge(alpha=1.0).fit(Xs(feats, tr_idx), ytr)
    models["ridge_baseline"] = (ridge, lambda f, i, m=ridge: np.clip(m.predict(Xs(f, i)), 0, RUL_CAP))

    try:
        from xgboost import XGBRegressor
        gb = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=seed, n_jobs=-1)
        gb_name = "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed)
        gb_name = "hist_gradient_boosting"
    gb.fit(Xs(feats, tr_idx), ytr)
    models[gb_name] = (gb, lambda f, i, m=gb: np.clip(m.predict(Xs(f, i)), 0, RUL_CAP))

    if use_lstm:
        models["lstm"] = (None, train_lstm(feats, tr_idx, seed))

    results = {}
    for name, (_, pred) in models.items():
        with run_ctx(name):
            yv, pv = feats.loc[val_idx, "rul"].to_numpy(), pred(feats, val_idx)
            pt = pred(test_feats, te_idx)
            r = {"val_rmse": rmse(yv, pv), "test_rmse": rmse(y_te, pt),
                 "test_nasa_score": nasa_score(y_te, pt), "test_cost": maintenance_cost(y_te, pt)}
            results[name] = r
            log({"model": name, "seed": seed, "window": 10}, r)
            print(f"{name:24s}", {k: round(v, 2) for k, v in r.items()})

    # Serving model = best *tabular* model by VALIDATION RMSE (never selected on the test set).
    serve = min([m for m in results if m in ("xgboost", "hist_gradient_boosting")],
                key=lambda m: results[m]["val_rmse"])
    model, pred = models[serve]

    # Conformal calibration on held-out engines
    q = conformal_quantile(feats.loc[cal_idx, "rul"], pred(feats, cal_idx), ALPHA)
    pt = pred(test_feats, te_idx)
    cov = coverage(y_te, np.clip(pt - q, 0, RUL_CAP), np.clip(pt + q, 0, RUL_CAP))

    bundle = {
        "model": model, "scaler": scaler, "cols": cols, "name": serve, "conformal_q": q, "alpha": ALPHA,
        "val_rmse": results[serve]["val_rmse"],
        "reference": feats.loc[tr_idx, cols].sample(2000, random_state=seed).reset_index(drop=True),
    }
    joblib.dump(bundle, MODELS / "model.joblib")

    base = results["ridge_baseline"]
    best_name = min((m for m in results if m != "ridge_baseline"), key=lambda m: results[m]["val_rmse"])
    best = results[best_name]
    summary = {
        "results": results, "serving_model": serve, "best_by_val": best_name,
        "conformal_q": q, "interval_coverage_test": cov, "target_coverage": 1 - ALPHA,
        "rmse_reduction_pct": pct_reduction(base["test_rmse"], best["test_rmse"]),
        "cost_reduction_pct": pct_reduction(base["test_cost"], best["test_cost"]),
    }
    (ROOT / "results.json").write_text(json.dumps(summary, indent=2))
    if mlflow:
        with run_ctx("serving_model"):
            log(metrics={"conformal_coverage_test": cov,
                         "rmse_reduction_pct": summary["rmse_reduction_pct"],
                         "cost_reduction_pct": summary["cost_reduction_pct"]})
            log_serving_model(model, "model")

    print("\n=== NUMBERS FOR YOUR CV (from results.json) ===")
    print(f"Best model by validation: {best_name}")
    print(f"RMSE reduction vs ridge baseline: {summary['rmse_reduction_pct']:.1f}%")
    print(f"Expected maintenance-cost reduction: {summary['cost_reduction_pct']:.1f}%")
    print(f"90% conformal interval coverage on test: {cov*100:.1f}% (interval +/- {q:.1f} cycles)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.lstm, a.seed)