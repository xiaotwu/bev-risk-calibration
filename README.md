# Stopping-Support Bias and Progress Costs Under Stale Bird's-Eye-View Observations

[Read the final IEEE report](paper/stale-bev-braking-risk.pdf)

The report follows a stale lead-vehicle observation from BEV tracking through the braking decision to closed-loop travel. It combines a collision audit, matched trajectory replay, and paired closed-loop simulation. A short discussion connects observation quality to the validation of driving-risk indicators.

## Reproducibility

[Evidence tables](evidence/README.md) · [Risk controller](risk/README.md) · [Trajectory replay](replay/README.md) · [Layer-Gate source overlay](layer_gate/README.md) · [Rights and attribution](RIGHTS.md)

From the repository root with Python 3.10 or newer, install the dependencies and run the archived evidence and controller checks.

```sh
python -m pip install -r requirements.txt
python -B verify_release.py
```

The checks recompute the headline simulation results from frozen scene-level tables and exercise the portable controller and replay code. The replay instructions describe how to acquire the original road and test-track data. The original CARLA capture and detector-training environments are outside the scope of the portable checks.
