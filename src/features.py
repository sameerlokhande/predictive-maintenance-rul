"""Rolling-window degradation features, computed per engine (no leakage across engines)."""
import pandas as pd
from .data import DROP, COLS

SENSORS = [c for c in COLS[2:] if c not in DROP]
WINDOW = 10


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must contain `unit`, `cycle` and raw sensor columns."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit")[SENSORS]
    mean = g.rolling(WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    std = g.rolling(WINDOW, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
    first = g.transform(lambda s: s.iloc[:WINDOW].mean())
    return pd.concat(
        [
            df[["unit", "cycle"]],
            df[SENSORS],
            mean.add_suffix("_mean"),
            std.add_suffix("_std"),
            (df[SENSORS] - first).add_suffix("_drift"),  # deviation from the engine's own early-life level
        ],
        axis=1,
    )


def feature_columns(feats: pd.DataFrame):
    return [c for c in feats.columns if c != "unit"]
