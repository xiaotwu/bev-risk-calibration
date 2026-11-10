"""Retrospective whole-log calibration for the V4 final R3 study.

The reserve targets max(g_est-g_ref, B_est-B_ref), with B denoting stopping
support. Calibration rows are selected-reference states after the fixed 2 s
warm-up, INCLUDING wrong selected leads. Each held-out log is excluded before
any parameter or training diagnostic is computed. Equal weight is assigned to
each training log with usable rows; empty logs remain visible in the roster.

The weighted empirical 90th percentile is a descriptive reserve, not a
conformal guarantee, collision probability, or a 10% action-risk budget. The
rows are temporally dependent and the method was designed using known data.
This module does not apply the reserve to a state or controller: downstream
code must apply it once, identically in each declared comparison arm.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence

ESTIMATORS = ("fresh", "mean", "envelope")
PRIMARY_QUANTILE_LEVEL = 0.90
WARMUP_S = 2.0


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def joint_optimism_error(g_est_m: float, B_est_m: float,
                         g_ref_m: float, B_ref_m: float) -> float:
    """Signed joint upper-error score; do not discard negative observations.

    If this score is <= q, subtracting q ONCE from both estimated g and B
    produces lower bounds for the corresponding reference quantities. Both
    constraints matter: B alone cannot certify a positive reference gap.
    Whether an empirical q covers a new state must be evaluated separately.
    """
    values = [_finite_number(v, name) for v, name in zip(
        (g_est_m, B_est_m, g_ref_m, B_ref_m),
        ("g_est_m", "B_est_m", "g_ref_m", "B_ref_m"))]
    g_est, B_est, g_ref, B_ref = values
    return max(g_est-g_ref, B_est-B_ref)


def weighted_inverse_cdf(values: Sequence[float], weights: Sequence,
                         level: float) -> float:
    """First observed value whose exact cumulative weight reaches level.

    There is no interpolation or finite-sample rank correction. Rational
    weights are retained exactly. Float weights/levels are interpreted from
    their decimal string representation, not their binary expansion.
    """
    alpha = _finite_number(level, "level")
    if not 0 < alpha <= 1:
        raise ValueError("level must be in (0, 1]")
    if len(values) != len(weights) or not len(values):
        raise ValueError("Nonempty aligned values and weights required")
    pairs = []
    for value, weight in zip(values, weights):
        val = _finite_number(value, "quantile value")
        if isinstance(weight, bool):
            raise ValueError("Quantile weights must be positive and finite")
        try:
            w = weight if isinstance(weight, Fraction) else Fraction(str(weight))
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("Quantile weights must be positive and finite") from exc
        if w <= 0:
            raise ValueError("Quantile weights must be positive")
        pairs.append((val, w))
    pairs.sort(key=lambda pair: pair[0])
    total = sum((w for _, w in pairs), Fraction(0, 1))
    target = Fraction(str(alpha))*total
    cumulative = Fraction(0, 1)
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= target:
            return value
    raise AssertionError("Positive exact weights failed to reach quantile")


def _validated_rows(rows: Iterable[Mapping], log_ids: Sequence[str]) -> list[dict]:
    allowed = set(log_ids)
    result, keys = [], set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every calibration row must be a mapping")
        log = row.get("log_id")
        estimator = row.get("estimator")
        if log not in allowed:
            raise ValueError(f"Calibration row has undeclared log_id: {log!r}")
        if estimator not in ESTIMATORS:
            raise ValueError(f"Unknown calibration estimator: {estimator!r}")
        timestamp = row.get("timestamp_ns")
        if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
            raise ValueError("timestamp_ns must remain an integer")
        key = (log, int(timestamp), estimator)
        if key in keys:
            raise ValueError(f"Duplicate calibration key: {key}")
        keys.add(key)
        error = _finite_number(row.get("error_m"), "error_m")
        if "elapsed_s" in row and _finite_number(row["elapsed_s"], "elapsed_s") < WARMUP_S:
            raise ValueError("Calibration input contains a pre-warm-up row")
        for flag in ("reference_available", "reference_has_lead",
                     "reference_velocity_valid", "selected_state"):
            if flag in row and row[flag] is not True:
                raise ValueError(f"Calibration input is not eligible: {flag}")
        wrong = row.get("wrong_lead")
        if wrong is not None and not isinstance(wrong, bool):
            raise ValueError("wrong_lead must be boolean or absent")
        result.append(dict(log_id=log, timestamp_ns=int(timestamp),
                           estimator=estimator, error_m=error, wrong_lead=wrong))
    return result


def fit_folds(rows: Iterable[Mapping], log_ids: Sequence[str],
              level: float = PRIMARY_QUANTILE_LEVEL) -> dict[str, dict]:
    """Fit one training-only reserve artifact for each held-out log.

    Required row keys: log_id, integer timestamp_ns, estimator (fresh/mean/
    envelope), finite signed error_m. Optional wrong_lead records a diagnostic
    and never excludes a row. Optional elapsed_s and eligibility flags are
    checked, rather than silently treating invalid observations as zero error.

    Every roster log appears in every appropriate training roster, including
    logs with no usable rows. Each estimator gives total weight 1 to every
    contributing training log and 1/n_l to each of that log's rows. The reserve
    is max(0, weighted_inverse_cdf(..., .90)). A completely empty training pool
    for any estimator is an error. Own-held-out row values and counts never
    appear in that fold's artifact, including its diagnostic counts.
    """
    alpha = _finite_number(level, "level")
    if alpha != PRIMARY_QUANTILE_LEVEL:
        raise ValueError("The frozen primary calibration level is 0.90")
    logs = list(log_ids)
    if len(logs) < 2 or any(not isinstance(log, str) or not log for log in logs):
        raise ValueError("At least two nonempty string log IDs are required")
    if len(set(logs)) != len(logs):
        raise ValueError("Calibration log roster must be unique")
    data = _validated_rows(rows, logs)
    folds = {}
    for heldout in logs:
        training = [log for log in logs if log != heldout]
        training_set = set(training)
        q_m, raw_quantiles, counts = {}, {}, {}
        for estimator in ESTIMATORS:
            selected = [row for row in data if row["log_id"] in training_set
                        and row["estimator"] == estimator]
            if not selected:
                raise ValueError(f"No valid calibration rows for {estimator!r} "
                                 f"when excluding held-out log {heldout!r}")
            by_log = Counter(row["log_id"] for row in selected)
            wrong_by_log = Counter(row["log_id"] for row in selected if row["wrong_lead"] is True)
            raw = weighted_inverse_cdf(
                [row["error_m"] for row in selected],
                [Fraction(1, by_log[row["log_id"]]) for row in selected], alpha)
            raw_quantiles[estimator] = raw
            q_m[estimator] = max(0.0, raw)
            counts[estimator] = dict(
                training_rows=len(selected),
                effective_log_count=len(by_log),
                contributing_log_ids=[log for log in training if by_log[log]],
                empty_training_log_ids=[log for log in training if not by_log[log]],
                rows_by_log={log: by_log[log] for log in training},
                wrong_lead_rows=sum(wrong_by_log.values()),
                wrong_lead_rows_by_log={log: wrong_by_log[log] for log in training},
                wrong_lead_flag_missing_rows=sum(row["wrong_lead"] is None for row in selected),
                negative_error_rows=sum(row["error_m"] < 0 for row in selected),
                total_log_weight=len(by_log))
        folds[heldout] = dict(
            heldout_log_id=heldout,
            training_log_ids=training,
            quantile_level=alpha,
            score="max(g_est-g_ref, B_est-B_ref)",
            weighting="Equal contributing-training-log weight; equal row weight within each log",
            quantile_rule="Exact weighted inverse CDF; first cumulative weight >= level; no interpolation",
            q_m=q_m,
            unclipped_quantiles_m=raw_quantiles,
            counts=counts,
            interpretation="Retrospective empirical reserve; no formal coverage or action-risk guarantee")
    return folds
