# Stale BEV Observations and Braking Risk

This research companion accompanies [Braking-Risk Bias and Progress Costs Under Stale Bird's-Eye-View Observations](paper/stale-bev-braking-risk.pdf), an English IEEE double-column manuscript. The editable LaTeX source and the published figure are in [paper](paper/).

The manuscript separates three evidence sources. The retrospective collision audit identifies saved states at which braking backup viability was lost. Paired closed-loop simulations quantify travel under feedback. Replays of measured road and test-track trajectories quantify open-loop estimation error during an imposed observation hold. The simulation holdouts support a descriptive comparison. Formal safety noninferiority was not tested. Layer-Gate is the upstream project context.

## Run the companion

Use Python 3.10 or newer. From the repository root:

```sh
python -m pip install -r requirements.txt
python -B verify_release.py
python -B -m risk
```

The risk-controller module and its checks are described in [risk](risk/README.md). [Replay](replay/README.md) documents how to obtain the original FHWA and JRC inputs and rerun the matched trajectory analysis. The [Layer-Gate source overlay](layer_gate/README.md) preserves the local camera/LiDAR fusion changes against a pinned upstream checkout.

## Reproducibility scope

The included evidence allows numerical readback of the paper's paired simulation travel and contact outcomes, historical collision-boundary counts, and common-history error summaries. It does not reproduce the original CARLA capture jobs or the full detector-training environment. The archive includes no private credentials, third-party binaries, model weights, point clouds, or raw road or test-track trajectories.

See [Evidence and provenance](evidence/README.md) and [Rights and attribution](RIGHTS.md).
