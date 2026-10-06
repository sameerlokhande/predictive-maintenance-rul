import numpy as np

# Asymmetric maintenance cost (per cycle of error). Override via arguments.
C_EARLY = 1.0   # premature replacement: prediction below true RUL, wasted life
C_LATE = 5.0    # missed-failure risk: prediction above true RUL


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def nasa_score(y, p):
    """Official C-MAPSS score: late predictions are penalised more than early ones."""
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def maintenance_cost(y, p, c_early=C_EARLY, c_late=C_LATE):
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.mean(np.where(d < 0, -d * c_early, d * c_late)))


def pct_reduction(baseline, new):
    return 100.0 * (baseline - new) / baseline
