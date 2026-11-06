# Recorded-trajectory replay

This package reproduces the open-loop trajectory component of the paper. The fixed two-second analysis compares early braking with matched early stable windows. Age sweeps cover one, two, and three seconds. The scripts estimate how stale motion observations affect stopping-support error while retaining measured trajectories.

## Offline check

Use Python 3.10 to 3.12 and an isolated virtual environment.

```bash
python -m pip install -r replay/requirements.txt
python replay/selfcheck.py
```

The self-check exercises synthetic braking and steady-speed trajectories through both scorers. It verifies matching, the stopping-support calculation, and parity at one second. It requires no dataset, model weights, GPU, or simulator.

## Official data

Run these commands from the repository root. Source acquisition and scoring are separate. Raw trajectories and generated row-level scores are excluded from the repository.

```bash
python replay/acquire_ngsim.py --workers 4
python replay/reconstruct_ngsim_recordings.py
python replay/replay_recordings.py
python replay/openacc/acquire_astazero.py
python replay/openacc/replay_openacc.py
```

- [FHWA NGSIM](https://doi.org/10.21949/1504477) provides the I-80 and US-101 records. The acquisition scripts download the original time windows from the official API and reconstruct the six recording-clock partitions, retaining the source queries and file hashes. The full acquisition writes several hundred MB and the API may throttle repeated requests.
- [JRC OpenACC](https://doi.org/10.2905/JRC.KMH3D00) provides the ten AstaZero platoon runs. The script records raw file hashes and obtains the provider's data attribution information.

The acquisition commands write beneath this directory. To score an existing download, pass `--data-root PATH` to either scoring command. For NGSIM, PATH must contain `raw_recordings/receipt.json` and the six referenced compressed recording files. For OpenACC, PATH must contain `openacc/raw/receipt.json` and the ten referenced compressed runs. Both scorers check the input inventory and file hashes. Use `--output PATH` to select a new output directory. The default outputs are `replay/results_recordings/` and `replay/openacc/results/`. The scorers preserve existing output directories by requiring a fresh destination for each run.

## Expected results

The two-second primary analysis contains 963 I-80 pairs, 1,275 US-101 pairs, and 22 AstaZero pairs. [Reference aggregates](reference_results.json) contain site and cohort means from the archived study. For NGSIM, select the two-second `position` rows under `by_site` in the generated summary. For AstaZero, select the two-second `matched` entries under `descriptive_window_pool` in the generated results.

This release preserves the study's eligibility, matching, and scoring algorithms. Its portable entry points validate inputs using acquisition receipts and log the released source hash. The study's archived preregistration locks used historical directory layouts. Numerical results and row-level output hashes were checked against the retained study inputs during release preparation.

These results quantify stopping-support error on recorded trajectories. BEV detection and deployed closed-loop safety require separate evidence.
