"""Generate a SYNTHETIC C-MAPSS-format dataset to smoke-test the pipeline (NOT for reported results)."""
import numpy as np, pandas as pd
from pathlib import Path

def make(out: Path, seed=0, n_train=100, n_test=100):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    trend = rng.uniform(-1, 1, 21) * (rng.random(21) < 0.5)
    base = rng.uniform(100, 600, 21)
    def engine(u, life, cut=None):
        T = life if cut is None else cut
        c = np.arange(1, T + 1)
        deg = (c / life) ** 2
        s = base + np.outer(deg, trend * 20) + rng.normal(0, 0.5, (T, 21))
        return np.column_stack([np.full(T, u), c, rng.normal(0, .002, T), rng.normal(0, .0003, T), np.full(T, 100.0), s])
    tr = np.vstack([engine(u, int(rng.integers(130, 360))) for u in range(1, n_train + 1)])
    lives = rng.integers(130, 360, n_test)
    cuts = [int(rng.integers(30, l - 1)) for l in lives]
    te = np.vstack([engine(u, int(l), c) for u, (l, c) in enumerate(zip(lives, cuts), 1)])
    rul = [int(l - c) for l, c in zip(lives, cuts)]
    pd.DataFrame(tr).to_csv(out / "train_FD001.txt", sep=" ", header=False, index=False)
    pd.DataFrame(te).to_csv(out / "test_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(rul).to_csv(out / "RUL_FD001.txt", index=False, header=False)

if __name__ == "__main__":
    make(Path(__file__).resolve().parents[1] / "data" / "raw")
