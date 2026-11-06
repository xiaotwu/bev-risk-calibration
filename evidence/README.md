# Evidence and provenance

| Evidence | Files | Unit of analysis |
|---|---|---|
| Retrospective collision analysis | `case_index.csv`, `control_timeline.csv`, `fixed_age_stress_boundaries.csv` | Selected simulated scene, trajectory, or decision |
| Independent closed loop | `diagnostic_pairs.csv`, `stress_pairs.csv`, `simulation_summary.json` | Paired held-out simulation scene |
| Common-history recalculation | `diagnostic_history_r4_scenes.csv`, `diagnostic_history_r4_stress_scenes.csv` | Scene within one source history pool |
| Recorded trajectories | `real_trajectory_summary.json` and the [replay implementation](../replay/) | Matched window, source recording, or test run |

Simulation tables come from the archived V5 R4 study. Historical collision decisions come from the prior V4 held-out comparison. The scenario indexes are simulator identifiers. Counts and outcomes are descriptive, and the source history pools share the same scene sets. The official road and test-track data are described in the replay documentation.

The archived manuscript PDF has SHA-256 `ea329fed85a9e09f82ae71953c77f1a1a9b2bab79d2e33dc8d8e56ec2f82f832`. Run `python -B evidence/verify_evidence.py` from the repository root to regenerate key table entries from the retained evidence.

