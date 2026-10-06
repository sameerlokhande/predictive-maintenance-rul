# Risk Register
| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Late prediction causes an unplanned failure | Medium | High | Asymmetric cost metric, conformal upper/lower bounds, human sign-off | ML engineer |
| R2 | Sensor drift or recalibration degrades accuracy | Medium | High | KS/PSI drift monitor, retraining trigger | ML engineer |
| R3 | Data leakage inflates reported results | Medium | Medium | Engine-level splits, model selection on validation only, leakage unit test | ML engineer |
| R4 | Model applied to a different fault mode or operating regime | Medium | High | Documented scope in model card; retrain per regime | Project lead |
| R5 | Untracked model change breaks the service | Low | Medium | MLflow registry, versioned CHANGELOG, Docker image tags | ML engineer |
