# Model Card - RUL Predictor
**Intended use:** rank turbofan engines by estimated remaining life to schedule maintenance. Decision support, not autonomous control.
**Data:** NASA C-MAPSS FD001 (100 train engines, 100 test engines, one operating condition, one fault mode).
**Target:** RUL in cycles, capped at 125.
**Models:** Ridge baseline; gradient boosting (served); LSTM (benchmark). Features: rolling statistics per engine.
**Metrics:** RMSE, NASA score, asymmetric maintenance cost, 90% conformal interval coverage (see results.json).
**Limitations:** simulated data; a single fault mode and operating condition (FD002-FD004 not covered); predictions
before cycle ~30 are unreliable; intervals assume calibration engines resemble deployment engines.
**Ethical / safety notes:** a late prediction is the costly error, so cost and NASA score are tracked alongside RMSE.
A human must approve maintenance decisions.
**Monitoring:** KS + PSI drift on features; retrain if >30% of features drift or live RMSE > 1.25x validation RMSE.
