"""Split conformal prediction intervals for regression."""
import numpy as np


def conformal_quantile(y_cal, p_cal, alpha=0.1):
    r = np.abs(np.asarray(y_cal, float) - np.asarray(p_cal, float))
    n = len(r)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(r, level, method="higher"))


def coverage(y, lo, hi):
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))
