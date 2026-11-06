# Evidence and provenance

| Evidence | Frozen files | Unit of analysis |
|---|---|---|
| Collision audit | `case_index.csv`, `boundary_timeline.csv`, `fixed_age_stress_boundaries.csv` | Selected simulated scene, trajectory, or braking decision |
| Closed-loop outcomes | `diagnostic_pairs.csv`, `stress_pairs.csv`, `simulation_summary.json` | Paired held-out simulation scene |
| Common-history recalculation | `diagnostic_history_r4_scenes.csv`, `diagnostic_history_r4_stress_scenes.csv` | Scene within a source history pool |
| Recorded trajectories | `real_trajectory_summary.json`, [replay implementation](../replay/) | Matched window, recording, or test run |

Closed-loop and common-history tables come from the V5 R4 study. Collision decisions come from the earlier V4 comparison. The boundary timeline retains the 49 decisions analyzed in the report. The two history pools at each rate share their scene set.

Run `python -B evidence/verify_evidence.py` from the repository root to check the primary simulation numbers.
