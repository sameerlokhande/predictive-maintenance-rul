"""Load NASA C-MAPSS FD001 and build RUL targets."""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
COLS = ["unit", "cycle"] + [f"setting{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
# Constant / near-constant channels in FD001 carry no degradation signal.
DROP = ["setting3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]
RUL_CAP = 125  # piece-wise linear RUL target: engines are "healthy" early in life


def _read(name: str) -> pd.DataFrame:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download C-MAPSS (FD001) and place train_FD001.txt, "
            "test_FD001.txt and RUL_FD001.txt in data/raw/ (see README)."
        )
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def load_train() -> pd.DataFrame:
    df = _read("train_FD001.txt")
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=RUL_CAP)
    return df


def load_test():
    """Returns (test sensor history, true RUL at the last observed cycle per engine)."""
    df = _read("test_FD001.txt")
    rul = pd.read_csv(RAW / "RUL_FD001.txt", sep=r"\s+", header=None, names=["rul"])
    rul.index = range(1, len(rul) + 1)
    return df, rul["rul"].clip(upper=RUL_CAP)
