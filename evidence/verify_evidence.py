"""Recompute the manuscript's principal simulation results from archived tables."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def rows(name):
    with (HERE / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def verify_simulation():
    summary = json.loads((HERE / "simulation_summary.json").read_text(encoding="utf-8"))
    result = {}
    for role, expected in summary.items():
        table = rows(f"{role}_pairs.csv")
        n = len(table)
        assert n == expected["n"]
        assert sorted(int(row["scene"]) for row in table) == list(range(n))
        travel = np.asarray([float(row["corrected_travel_m"]) - float(row["reference_travel_m"])
                             for row in table])
        pairs = Counter((int(row["reference_contact"]), int(row["corrected_contact"])) for row in table)
        assert pairs == {(0, 0): n}
        assert expected["contact_discordance"] == dict(candidate_only=0, reference_only=0, both=0, neither=n)
        estimate = float(travel.mean())
        indices = np.random.default_rng(1597391170).integers(0, n, size=(4000, n))
        interval = np.quantile(travel[indices].mean(axis=1), [.025, .975])
        assert math.isclose(estimate, expected["travel_difference_m"], rel_tol=1e-12)
        np.testing.assert_allclose(interval, expected["descriptive_bootstrap_95_m"], rtol=1e-12, atol=1e-12)
        bound = 1.0 - .025 ** (1.0 / n)
        assert math.isclose(bound, expected["conservative_descriptive_upper"], rel_tol=1e-12)
        result[role] = dict(pairs=n, mean_travel_difference_m=estimate,
                            interval_m=interval.tolist(), contact_rate_upper=bound)
    return result


def verify_forensics():
    cases = rows("case_index.csv")
    boundaries = rows("fixed_age_stress_boundaries.csv")
    assert len(cases) == 62 and len({row["index"] for row in cases}) == 31
    assert sum(row["collision"] == "True" for row in cases) == 51
    assert len(boundaries) == 49
    assert sum(row["command_changed"] == "True" for row in boundaries) == 35
    assert sum(row["fixed_backup_remains_viable"] == "True" for row in boundaries) == 32
    return dict(scenes=31, contact_trajectories=51, identifiable_boundaries=49,
                changed_commands=35, viable_fixed_backups=32)


def verify_common_histories():
    result = {}
    cohorts = {
        "diagnostic_history_r4_scenes.csv": {
            "reference_history": (290, 304, 3197, -5.448, 3.542),
            "candidate_history": (290, 457, 3216, -5.513, 3.542),
        },
        "diagnostic_history_r4_stress_scenes.csv": {
            "reference_history": (74, 21, 1634, -5.832, 1.671),
            "candidate_history": (74, 45, 1634, -5.796, 1.684),
        },
    }
    for filename, expected in cohorts.items():
        table = rows(filename)
        for pool, targets in expected.items():
            supported = [row for row in table if row["pool"] == pool and int(row["common_support"]) > 0]
            changes = sum(int(row["same_epsilon_changes"]) for row in supported)
            decisions = sum(int(row["action_support"]) for row in supported)
            assert changes == sum(int(row["fixed_backup_both_viable"]) for row in supported)
            assert changes == sum(int(row["same_epsilon_lower_commands"]) for row in supported)
            metrics = []
            for suffix in ("max_positive_m", "mean_negative_m"):
                change = [float(row[f"stress_{suffix}"]) - float(row[f"cv_{suffix}"]) for row in supported]
                metrics.append(round(float(np.mean(change)), 3))
            actual = (len(supported), changes, decisions, *metrics)
            assert actual == targets, (filename, pool, actual, targets)
            result[f"{filename}:{pool}"] = actual
    return result


if __name__ == "__main__":
    print(json.dumps(dict(status="PASS", simulation=verify_simulation(),
                          forensics=verify_forensics(), common_history=verify_common_histories()), indent=2))

