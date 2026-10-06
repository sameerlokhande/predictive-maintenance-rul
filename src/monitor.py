"""Data-drift and performance monitoring with retraining triggers (KS test + PSI)."""
import numpy as np
from scipy.stats import ks_2samp

PSI_ALERT = 0.2               # rule of thumb: PSI > 0.2 = significant shift
KS_P_ALERT = 0.01
DRIFT_SHARE_ALERT = 0.3       # retrain if >30% of monitored features drift
RMSE_DEGRADATION_ALERT = 1.25 # retrain if live RMSE > 1.25x validation RMSE


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref_df, cur_df, cols):
    rows = []
    for c in cols:
        p_ks = float(ks_2samp(ref_df[c], cur_df[c]).pvalue)
        p_psi = psi(ref_df[c].to_numpy(), cur_df[c].to_numpy())
        rows.append({"feature": c, "ks_p": p_ks, "psi": p_psi,
                     "drift": bool(p_ks < KS_P_ALERT and p_psi > PSI_ALERT)})
    share = float(np.mean([r["drift"] for r in rows]))
    return {"features": rows, "drift_share": share, "retrain": share > DRIFT_SHARE_ALERT}


def performance_alert(val_rmse, live_rmse):
    return live_rmse > RMSE_DEGRADATION_ALERT * val_rmse
