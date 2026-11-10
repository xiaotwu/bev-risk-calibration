# Calibrating BEV Evidence for Longitudinal Risk Decisions

[Read the final IEEE-format paper](paper/BEV_Risk_IEEE_Final.pdf) · Xiaotong Wu · xiaotongwu116@gmail.com

This independent research project connects camera–LiDAR evidence in bird's-eye view (BEV) to longitudinal risk decisions. Its main method, **C + qF**, combines fixed object-level fusion and causal tracking with an empirical reserve calibrated on joint gap and stopping-support error. **R3** is an exploratory temporal evidence mechanism.

Retrospective evaluation covers ten previously inspected Argoverse 2 logs, eleven configurations, and four conditions. At recorded speeds, calibrated C reduces model-relative false admissions from **7/949 to 2/949**, with a **0.01849 m/s²** increase in mean positive command shortfall. A fixed 1 m reserve is more economical at equal error counts in the 12 m/s probes. R3 does not consistently improve the calibrated tradeoff. Main-method selection was informed by outcomes; these results do not establish independent validation, formal coverage, or improved driving safety.

## Contents

- `bev_risk/`: perception geometry, tracking, stopping constraints, evaluation, and AV2 data adapters.
- `bev_risk_v2/`: shared observation handling and fixed-fusion decision components used by the final method.
- `bev_risk_v4/method.py`: gap/support evidence pairs, temporal fusion, and reserve-aware decisions.
- `analysis_v4/calibration.py`: joint residuals and whole-log-excluded empirical calibration.
- `experiment_config*.json`: original parameter configurations.
- `tests/`: synthetic component checks.
- `results/`: all 44 final operating points and ten held-log reserve summaries.
- `paper/`: the final nine-page paper.

Scientific source files are unchanged. Tests have been grouped into one directory; historical module names remain to preserve imports and dependencies.

## Run the component checks

Use Python 3.12:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
```

All 63 included checks passed in this compact checkout. They require no sensor downloads or model weights. `requirements-perception.txt` adds optional camera-inference and Feather-data dependencies; installing them alone does not supply the inputs.

## Reproduction scope

This compact repository provides the core implementation, synthetic checks, published summary tables, and paper. It is **not a complete empirical replay package**. Data adapters require external sensor inputs and manifests. Full replay also requires the author-maintained separate reproducibility archive, including frozen evidence caches, reference data, and provenance. Raw data, weights, historical reports, and host-specific runners are omitted. See [RIGHTS.md](RIGHTS.md) for attribution and reuse scope.
