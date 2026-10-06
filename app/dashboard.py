"""Streamlit dashboard.  Run: streamlit run app/dashboard.py"""
import numpy as np
import pandas as pd
import streamlit as st
from src.data import load_test
from src.features import build_features, SENSORS
from src.predict import predict_rul, bundle
from src.monitor import drift_report

st.set_page_config(page_title="Predictive Maintenance RUL", layout="wide")
st.title("Predictive Maintenance - Remaining Useful Life")
b = bundle()
test_raw, test_rul = load_test()

tab1, tab2 = st.tabs(["Predict RUL", "Drift monitoring"])

with tab1:
    unit = st.selectbox("Engine", sorted(test_raw["unit"].unique()))
    hist = test_raw[test_raw["unit"] == unit].sort_values("cycle")
    res = predict_rul(hist.drop(columns=["unit"]).to_dict("records"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted RUL (cycles)", f"{res['rul']:.0f}")
    c2.metric(f"{int(res['confidence']*100)}% interval", f"{res['lower']:.0f} - {res['upper']:.0f}")
    c3.metric("True RUL", f"{test_rul.loc[unit]:.0f}")
    st.line_chart(hist.set_index("cycle")[SENSORS[:6]])

with tab2:
    shift = st.slider("Simulated sensor offset (std devs)", 0.0, 3.0, 0.0, 0.1)
    cur = build_features(test_raw)
    cols = [c for c in b["cols"] if c in SENSORS]
    for c in cols[: len(cols) // 2]:
        cur[c] = cur[c] + shift * b["reference"][c].std()
    rep = drift_report(b["reference"], cur, cols)
    st.metric("Drifted features", f"{rep['drift_share']:.0%}")
    st.error("Retraining trigger FIRED") if rep["retrain"] else st.success("No retraining needed")
    st.dataframe(pd.DataFrame(rep["features"]).sort_values("psi", ascending=False))
