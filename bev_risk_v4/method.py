"""Calibrated fresh/prior decision evidence, without a fabricated velocity.

For a physical estimate, let B = g + max(v,0)^2/16. The fixed stopping
contract has clearance M(a) = g - max(0,E(a)-max(v,0)^2/16)
                             = min(g, B-E(a)).
Taking coordinatewise minima of two evidence pairs therefore gives exactly
min(M_fresh(a), M_prior(a)) for every action. This is a conservative evidence
envelope, not a new physical velocity or a claim that either track is correct.

If e* = max(g_hat-g_ref, B_hat-B_ref) <= q, then
min(g_hat-q, B_hat-q-E(a)) <= min(g_ref, B_ref-E(a)) simultaneously for every
action and speed query. This implication is conditional on the error event;
an empirical calibration quantile does not itself establish a probability or
driving-safety guarantee. Calibration and reference labels live elsewhere.
"""
from __future__ import annotations

import copy
import math
from numbers import Integral

from bev_risk.policy import ACTIONS, nominal_action, stopping_distance, stress_shift
from bev_risk_v2.method import decide as legacy_decide


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Finite {name} required")
    return value


def _timestamp(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Source timestamp must be an integer number of nanoseconds")
    return int(value)


def _pair_values(pair):
    gap = _finite(pair["gap_m"], "evidence gap")
    support = _finite(pair["support_m"], "evidence support")
    if support < gap:
        raise ValueError("Stopping support cannot be smaller than its evidence gap")
    return gap, support


def pair_from_state(state, allowance_s=.2):
    """Apply the unchanged age stress, retaining the actual observation source.

    Missing/uninitialized states produce no pair. The output does not contain
    an inferred effective velocity: a subsequently fused pair can have its two
    coordinates supplied by different estimates.
    """
    if state is None or not state.get("initialized", True):
        return None
    gap = _finite(state["gap_m"], "state gap")
    speed = _finite(state["lead_speed_mps"], "state speed")
    age = _finite(state["age_s"], "observation age")
    if age < 0:
        raise ValueError("Observation age cannot be negative")
    if allowance_s is not None:
        allowance_s = _finite(allowance_s, "age allowance")
        if allowance_s < 0:
            raise ValueError("Age allowance cannot be negative")
    timestamp = _timestamp(state["last_update_ns"])
    corrected_gap, corrected_speed = stress_shift(gap, speed, age, allowance_s)
    source = dict(track_id=state["track_id"], last_update_ns=timestamp,
                  last_candidate_id=state.get("last_candidate_id"), age_s=age,
                  observations=int(state.get("observations", 0)),
                  updated=bool(state.get("updated", False)), initialized=True)
    return dict(gap_m=corrected_gap,
                support_m=corrected_gap+corrected_speed**2/16.,
                raw_gap_m=gap, raw_support_m=gap+max(0., speed)**2/16.,
                age_active=allowance_s is not None and age > allowance_s,
                correction_nonzero=(abs(corrected_gap-gap) > 1e-9 or
                                    abs(corrected_speed-max(0., speed)) > 1e-9),
                allowance_s=allowance_s, kind="state", source=source)


def valid_prior(fresh_state, prior_state):
    """Check source/identity consistency of a runner-constructed one-step prior.

    The runner must obtain the prior from the immediately previous C-tracker
    snapshot, propagated to the same decision time. State dictionaries cannot
    prove snapshot provenance. Source age is allowed to exceed one frame during
    missing observations; it must never be relabeled as a fresh observation.
    Estimated track-ID equality is not ground-truth identity certification.
    """
    if fresh_state is None or prior_state is None:
        return False
    try:
        if not (fresh_state.get("initialized", True) and
                prior_state.get("initialized", True) and
                fresh_state.get("updated", False)):
            return False
        fresh_id, prior_id = fresh_state["track_id"], prior_state["track_id"]
        if fresh_id is None or prior_id is None or fresh_id != prior_id:
            return False
        fresh_time = _timestamp(fresh_state["last_update_ns"])
        prior_time = _timestamp(prior_state["last_update_ns"])
        fresh_age = _finite(fresh_state["age_s"], "fresh age")
        prior_age = _finite(prior_state["age_s"], "prior age")
        if fresh_age != 0. or prior_age < 0. or prior_time >= fresh_time:
            return False
        # Subtract integer timestamps before conversion; never round a ~1e17 ns
        # timestamp through a float. Both states must describe the same query.
        elapsed = (fresh_time-prior_time)*1e-9
        if not math.isclose(prior_age-fresh_age, elapsed, rel_tol=0., abs_tol=1e-9):
            return False
        _finite(fresh_state["gap_m"], "fresh gap")
        _finite(prior_state["gap_m"], "prior gap")
        _finite(fresh_state["lead_speed_mps"], "fresh speed")
        _finite(prior_state["lead_speed_mps"], "prior speed")
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def fuse_pairs(fresh, prior, kind="envelope"):
    """Combine valid evidence pairs; exact ties retain both possible sources.

    A caller must pass prior=None when valid_prior fails. Missing fresh evidence
    stays missing: this function never creates a selected object from a prior
    alone. Mean fusion is a fixed equal-weight control, not a fitted estimate.
    """
    if kind not in ("envelope", "mean"):
        raise ValueError("Fusion kind must be envelope or mean")
    if fresh is None:
        return None
    fresh_values = _pair_values(fresh)
    keys = ("gap_m", "support_m")
    if prior is None:
        result = copy.deepcopy(fresh)
        result.update(kind="fresh_only", fusion_requested=kind, used_prior=False,
                      coordinate_sources={k: ["fresh"] for k in keys},
                      coordinate_ties={k: False for k in keys},
                      coordinate_weights={k: {"fresh": 1.} for k in keys},
                      sources={"fresh": copy.deepcopy(fresh["source"])})
        return result
    prior_values = _pair_values(prior)
    result = dict(kind=kind, fusion_requested=kind, used_prior=True,
                  age_active=bool(fresh.get("age_active") or prior.get("age_active")),
                  correction_nonzero=bool(fresh.get("correction_nonzero") or
                                          prior.get("correction_nonzero")),
                  sources={"fresh": copy.deepcopy(fresh["source"]),
                           "prior": copy.deepcopy(prior["source"])},
                  coordinate_sources={}, coordinate_ties={}, coordinate_weights={})
    for key, f, p in zip(keys, fresh_values, prior_values):
        result["coordinate_ties"][key] = f == p
        if kind == "mean":
            result[key] = .5*f+.5*p
            result["coordinate_sources"][key] = ["fresh", "prior"]
            result["coordinate_weights"][key] = dict(fresh=.5, prior=.5)
        else:
            result[key] = min(f, p)
            contributors = [name for name, value in (("fresh", f), ("prior", p))
                            if value == result[key]]
            result["coordinate_sources"][key] = contributors
            # A tie has two sufficient contributors, not an invented unique
            # source or a claimed physical averaging weight.
            result["coordinate_weights"][key] = (
                {contributors[0]: 1.} if len(contributors) == 1 else None)
    _pair_values(result)
    return result


def decide_pair(pair, ego_speed, evidence, buffer_m=0., free_guard_m=1.):
    """Use a calibrated lower margin for tracked evidence; common fallback stays.

    q applies only to tracked evidence. No selected pair uses the identical V2
    finite-prefix/unknown fallback with buffer=0, even if the fitted tracked
    error allowance is positive. This prevents calibration from silently
    changing missing-object handling between estimators.
    """
    speed = _finite(ego_speed, "ego speed")
    if speed < 0:
        raise ValueError("Ego speed cannot be negative")
    buffer = _finite(buffer_m, "calibration allowance")
    if buffer < 0:
        raise ValueError("Calibration allowance cannot be negative")
    if pair is None:
        decision = legacy_decide(None, speed, evidence, allowance_s=None,
                                 buffer_m=0., repair=True, free_guard_m=free_guard_m)
        return decision | dict(gap_m=None, buffer_m=0.,
                               requested_buffer_m=buffer,
                               calibration_applied=False,
                               evidence_gap_m=None, evidence_support_m=None,
                               fusion_kind=None)
    gap, support = _pair_values(pair)
    lower_gap, lower_support = gap-buffer, support-buffer
    nominal = nominal_action(speed)
    margins = [dict(action=a, margin_m=min(lower_gap, lower_support-stopping_distance(speed, a)),
                    at_or_below_nominal=a <= nominal) for a in ACTIONS]
    admitted = [row["action"] for row in margins
                if row["at_or_below_nominal"] and row["margin_m"] > 0.]
    return dict(action=max(admitted) if admitted else -6., nominal_action=nominal,
                reason="admitted" if admitted else "full_brake_fallback",
                gap_m=lower_gap, corrected_gap_m=lower_gap, support_m=lower_support,
                corrected_speed_mps=None, evidence_gap_m=gap,
                evidence_support_m=support, buffer_m=buffer,
                requested_buffer_m=buffer, calibration_applied=buffer > 0.,
                age_active=bool(pair.get("age_active", False)),
                correction_nonzero=bool(pair.get("correction_nonzero", False)),
                observation_state="tracked", free_space_used=False,
                fusion_kind=pair.get("kind", "state"), action_margins=margins)
