"""Simulate sensor drift on test data and show the monitor firing a retraining trigger."""
import numpy as np
from src.data import load_test
from src.features import build_features, SENSORS
from src.monitor import drift_report
from src.predict import bundle

b = bundle()
test_raw, _ = load_test()
cur = build_features(test_raw)
ref = b["reference"]
cols = [c for c in b["cols"] if c in SENSORS]  # raw sensor channels

print("== No drift (real test data) ==")
r0 = drift_report(ref, cur, cols)
print(f"drift share: {r0['drift_share']:.0%} -> retrain: {r0['retrain']}")

print("\n== Simulated sensor drift (+1.5 std offset on half of the channels) ==")
shifted = cur.copy()
for c in cols[: len(cols) // 2]:
    shifted[c] = shifted[c] + 1.5 * ref[c].std()
r1 = drift_report(ref, shifted, cols)
print(f"drift share: {r1['drift_share']:.0%} -> retrain: {r1['retrain']}")
print([x["feature"] for x in r1["features"] if x["drift"]])
