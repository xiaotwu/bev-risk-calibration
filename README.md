# Predictive Value of Observation Age for Stopping-Support Error in Longitudinal Braking

[Manuscript PDF](paper/observation-age/observation-age-stopping-support-paper.pdf) · [Supporting results](paper/observation-age/REPRODUCE.md)

The manuscript evaluates observation age as a predictor of stopping-support error on four held-out trajectory campaigns. It compares predictions by horizon, tests horizon-specific calibration and fitting, and separately evaluates paired bird's-eye-view detector outputs near a prespecified braking boundary.

## Earlier Report and Reproducibility

[Earlier report](paper/stale-bev-braking-risk.pdf) · [Evidence tables](evidence/README.md) · [Risk controller](risk/README.md) · [Trajectory replay](replay/README.md) · [Layer-Gate source overlay](layer_gate/README.md) · [Rights and attribution](RIGHTS.md)

From the repository root with Python 3.10 or newer, install the dependencies and run the archived evidence and controller checks.

```sh
python -m pip install -r requirements.txt
python -B verify_release.py
```

The checks recompute the headline simulation results from frozen scene-level tables and exercise the portable controller and replay code. The replay instructions describe how to acquire the original road and test-track data. The original CARLA capture and detector-training environments are outside the scope of the portable checks.
