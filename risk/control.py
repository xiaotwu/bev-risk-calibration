"""Shared delayed longitudinal controllers using perception beliefs only.

The contract independently copies the continuous-motion mathematics from the
preserved backup_contract.py: leader brakes at -8 immediately; ego executes its
pending command for .2 s, the candidate for .2 s, then -6 until rest. Speeds do
not reverse. Collision means gap <= 0 at ANY continuous time, including now.
The final braking segment is evaluated analytically instead of in .2 s chunks.

Beliefs have mean_gap, mean_vlead, sigma_gap, sigma_v, presence. Optional rho
or cov/covariance (gap, leader-speed order) supplies correlation; the supplied
sigmas always define marginal uncertainty. Optional has_track=False requests a
conservative RSS fallback. Missing/nonfinite state means trigger full braking.
No GT, simulator API, learned model, random generator or file access is used.

Risk is presence times the conditional collision frequency on FIXED 128-point
antithetic normal Halton quadrature. It is a commitment-risk estimate, not an
episode-safety or calibrated-probability guarantee. RSS evaluates the SAME full
backup trajectory at gap-k*sigma_gap and max(0,vlead-k*sigma_v). TTC is the
minimum positive-closing TTC over both commitment stages, including continuous
interior extrema and stopping events; it rejects a candidate when TTC < its
threshold. This is a stronger delay-aware TTC baseline than current-TTC only.
All methods select the largest member of ACTIONS <= shared nominal command.
No admissible action means -6, which itself is not a safety certificate.
"""
from __future__ import annotations

from collections.abc import Mapping
from statistics import NormalDist
import numpy as np

DT = 0.2
LEADER_BRAKE = 8.0
EGO_BACKUP_BRAKE = 6.0
MIN_COMMAND = -6.0
MAX_COMMAND = 3.0
ACTIONS = np.array([-6.0, -3.0, 0.0, 1.5, 3.0], dtype=np.float64)
ACTIONS.setflags(write=False)
TTC_THRESHOLDS = (0.0, .25, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, np.inf)
QMC_SAMPLES = 128
FALLBACK_RSS_K = 3.0
SEMANTICS = "pending_0.2s_then_candidate_0.2s_then_ego_brake6_leader_brake8"


def _halton(index, base):
    value, factor = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


_normal = NormalDist()
_base_normal = np.array([
    [_normal.inv_cdf(_halton(i, 2)), _normal.inv_cdf(_halton(i, 3))]
    for i in range(1, QMC_SAMPLES // 2 + 1)
], dtype=np.float64)
NORMAL_QMC = np.concatenate((_base_normal, -_base_normal), axis=0)
NORMAL_QMC.setflags(write=False)
del _base_normal, _normal


def displacement(speed, acceleration, duration):
    """Exact constant-acceleration displacement with absorbing zero speed."""
    v, a, h = np.broadcast_arrays(*[
        np.asarray(x, dtype=np.float64) for x in (speed, acceleration, duration)
    ])
    stopping_time = np.divide(v, -a, out=np.full(v.shape, np.inf), where=a < 0)
    active_time = np.minimum(h, stopping_time)
    return v * active_time + .5 * a * active_time ** 2


def advance(gap, ego_speed, leader_speed, ego_acceleration,
            leader_acceleration=-LEADER_BRAKE, dt=DT):
    """Next states and exact continuous minimum gap, including stop events."""
    g, ve, vl, ae, al, h = np.broadcast_arrays(*[
        np.asarray(x, dtype=np.float64)
        for x in (gap, ego_speed, leader_speed, ego_acceleration, leader_acceleration, dt)
    ])
    gn = g + displacement(vl, al, h) - displacement(ve, ae, h)
    relative_acceleration = al - ae
    stationary_time = np.divide(
        ve - vl, relative_acceleration, out=np.full(g.shape, np.inf),
        where=np.abs(relative_acceleration) > 1e-14)
    ego_stop = np.divide(ve, -ae, out=np.full(g.shape, np.inf), where=ae < 0)
    leader_stop = np.divide(vl, -al, out=np.full(g.shape, np.inf), where=al < 0)
    interior = ((stationary_time > 0) & (stationary_time < h)
                & (stationary_time < ego_stop) & (stationary_time < leader_stop))
    t = np.where(interior, stationary_time, 0.0)
    stationary_gap = g + displacement(vl, al, t) - displacement(ve, ae, t)
    minimum_gap = np.minimum(np.minimum(g, gn), np.where(interior, stationary_gap, np.inf))
    return gn, np.maximum(0.0, ve + ae * h), np.maximum(0.0, vl + al * h), minimum_gap


def _validated_samples(samples):
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim < 2 or x.shape[-1] != 5 or x.shape[-2] == 0:
        raise ValueError("Expected nonempty [S,5] or [...,S,5] samples")
    if not np.isfinite(x).all() or np.any(x[..., 1:3] < 0):
        raise ValueError("Samples must be finite, with nonnegative speeds")
    if np.any(x[..., 3:5] < MIN_COMMAND) or np.any(x[..., 3:5] > MAX_COMMAND):
        raise ValueError("Pending and candidate commands must lie in [-6,3]")
    return x


def backup_outcomes(samples):
    """Exact per-sample backup outcomes; sample shape [S,5] or [...,S,5]."""
    x = _validated_samples(samples)
    g, ve, vl, pending, candidate = [x[..., i] for i in range(5)]
    g1, ve1, vl1, min1 = advance(g, ve, vl, pending)
    g2, ve2, vl2, min2 = advance(g1, ve1, vl1, candidate)
    duration = np.maximum(ve2 / EGO_BACKUP_BRAKE, vl2 / LEADER_BRAKE)
    terminal_gap, terminal_ve, terminal_vl, min3 = advance(
        g2, ve2, vl2, -EGO_BACKUP_BRAKE, dt=duration)
    minimum_gap = np.minimum(np.minimum(min1, min2), min3)
    return dict(collision=minimum_gap <= 0.0, minimum_gap_m=minimum_gap,
                terminal_gap_m=terminal_gap, terminal_ego_speed_mps=terminal_ve,
                terminal_leader_speed_mps=terminal_vl,
                first_step_gap_m=g1, first_step_ego_speed_mps=ve1,
                first_step_leader_speed_mps=vl1, second_step_gap_m=g2,
                second_step_ego_speed_mps=ve2, second_step_leader_speed_mps=vl2,
                propagation_duration_s=2 * DT + duration, dt_s=DT, semantics=SEMANTICS)


def backup_risk(samples, *, return_details=False):
    """Conditional collision frequency over the second-to-last sample axis."""
    details = backup_outcomes(samples)
    risk = np.mean(details["collision"], axis=-1)
    risk = float(risk) if risk.ndim == 0 else risk
    return (risk, details) if return_details else risk


def _minimum_ttc_in_step(gap, ego_speed, leader_speed, ego_acceleration, dt=DT):
    """Exact minimum TTC for positive closing over a constant-command stage.

    Split at either vehicle's stop, then solve d(gap/closing)/dt = 0
    analytically within each constant-relative-acceleration piece.
    """
    g, ve, vl, ae = np.broadcast_arrays(*[
        np.asarray(x, dtype=np.float64)
        for x in (gap, ego_speed, leader_speed, ego_acceleration)
    ])
    al = -LEADER_BRAKE
    te = np.minimum(dt, np.divide(ve, -ae, out=np.full(g.shape, np.inf), where=ae < 0))
    tl = np.minimum(dt, vl / LEADER_BRAKE)
    boundaries = np.sort(np.stack((np.zeros_like(g), te, tl, np.full_like(g, dt)), axis=-1), axis=-1)
    best = np.full(g.shape, np.inf)
    for i in range(3):
        start, end = boundaries[..., i], boundaries[..., i + 1]
        length = end - start
        gs = g + displacement(vl, al, start) - displacement(ve, ae, start)
        ves, vls = np.maximum(0., ve + ae * start), np.maximum(0., vl + al * start)
        a_active = np.where((ves > 1e-12) | (ae > 0), ae, 0.)
        l_active = np.where(vls > 1e-12, al, 0.)
        closing, b = ves - vls, a_active - l_active
        discriminant = -closing ** 2 - 2 * b * gs
        root_scale = np.sqrt(np.maximum(0., discriminant))
        roots = [np.divide(-closing + s * root_scale, b,
                           out=np.full(g.shape, np.inf), where=np.abs(b) > 1e-14)
                 for s in (-1., 1.)]
        for t, is_root in ((np.zeros_like(g), False), (length, False),
                           (roots[0], True), (roots[1], True)):
            valid = ((t >= 0) & (t <= length))
            if is_root:
                valid &= discriminant >= 0
            safe_t = np.where(valid, t, 0.)
            c = closing + b * safe_t
            gap_t = gs - closing * safe_t - .5 * b * safe_t ** 2
            value = np.divide(np.maximum(0., gap_t), c,
                              out=np.full(g.shape, np.inf), where=c > 1e-12)
            value = np.where(gap_t <= 0., 0., value)
            best = np.minimum(best, np.where(valid, value, np.inf))
    return best


def commitment_ttc(gap, ego_speed, leader_speed, pending_acc, candidate_acc):
    """Minimum continuous TTC over pending .2 s and candidate .2 s."""
    before = _minimum_ttc_in_step(gap, ego_speed, leader_speed, pending_acc)
    g1, ve1, vl1, _ = advance(gap, ego_speed, leader_speed, pending_acc)
    after = _minimum_ttc_in_step(g1, ve1, vl1, candidate_acc)
    return np.minimum(before, after)


def _get(belief, name, default=None):
    return belief.get(name, default) if isinstance(belief, Mapping) else getattr(belief, name, default)


def _belief_row(belief):
    means = np.array([_get(belief, "mean_gap", np.nan),
                      _get(belief, "mean_vlead", np.nan)], dtype=np.float64)
    sigmas = np.array([_get(belief, "sigma_gap", np.nan),
                       _get(belief, "sigma_v", np.nan)], dtype=np.float64)
    presence = float(_get(belief, "presence", 1.0))
    finite = np.isfinite(means).all() and np.isfinite(sigmas).all() and np.isfinite(presence)
    if not finite:
        return dict(valid=False, has_track=False, gap=0., vlead=0., sg=0., sv=0., presence=1., rho=0.)
    if np.any(sigmas < 0) or not 0 <= presence <= 1:
        raise ValueError("Belief sigmas must be nonnegative; presence must lie in [0,1]")
    rho = _get(belief, "rho", _get(belief, "correlation", None))
    cov = _get(belief, "cov", _get(belief, "covariance", None))
    if rho is None and cov is not None:
        cov = np.asarray(cov, dtype=np.float64)
        if cov.shape != (2, 2) or not np.isfinite(cov).all() or np.any(np.diag(cov) < 0):
            raise ValueError("Optional covariance must be finite 2x2 with nonnegative diagonal")
        if not np.allclose(cov, cov.T, rtol=1e-7, atol=1e-10):
            raise ValueError("Optional covariance must be symmetric")
        scale = np.sqrt(cov[0, 0] * cov[1, 1])
        if abs(cov[0, 1]) > scale + 1e-8:
            raise ValueError("Optional covariance must be positive semidefinite")
        rho = cov[0, 1] / scale if scale > 0 else 0.
    rho = 0. if rho is None else float(rho)
    if not np.isfinite(rho) or abs(rho) > 1. + 1e-8:
        raise ValueError("Correlation must lie in [-1,1]")
    return dict(valid=True, has_track=bool(_get(belief, "has_track", True)),
                gap=float(means[0]), vlead=float(means[1]),
                sg=float(sigmas[0]), sv=float(sigmas[1]), presence=presence,
                rho=float(np.clip(rho, -1., 1.)))


def _parameter(kind, param):
    if kind not in ("risk", "rss", "ttc"):
        raise ValueError("kind must be 'risk', 'rss' or 'ttc'")
    p = float(param)
    if np.isnan(p) or p < 0 or (kind == "risk" and (not np.isfinite(p) or p > 1)):
        raise ValueError("Invalid controller parameter")
    if kind == "rss" and not np.isfinite(p):
        raise ValueError("RSS k must be finite")
    return p


def _motion_inputs(ego_speed, pending_acc, candidates):
    ve, pending = float(ego_speed), float(pending_acc)
    candidates = np.atleast_1d(np.asarray(candidates, dtype=np.float64))
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("candidates must be a nonempty one-dimensional sequence")
    if not np.isfinite(ve) or ve < 0 or not np.isfinite(pending):
        raise ValueError("Ego speed must be finite/nonnegative and pending finite")
    if (pending < MIN_COMMAND or pending > MAX_COMMAND or not np.isfinite(candidates).all()
            or np.any(candidates < MIN_COMMAND) or np.any(candidates > MAX_COMMAND)):
        raise ValueError("Commands must lie in [-6,3]")
    return ve, pending, candidates


def evaluate_candidates(belief, ego_speed, pending_acc, candidates=ACTIONS,
                        kind="risk", param=.01):
    """Vectorized candidate diagnostics, without nominal-action filtering.

    safe, risk, conditional_risk, minimum_gap_m, point_margin_m, rss_margin_m
    and ttc_s are one-dimensional arrays matching candidates. minimum_gap_m
    is the minimum over the fixed conditional belief samples, not a GT margin.
    Use evaluate_candidate for scalar/JSON-friendly selected-action records.
    """
    p = _parameter(kind, param)
    ve, pending, actions = _motion_inputs(ego_speed, pending_acc, candidates)
    b = _belief_row(belief)
    z = NORMAL_QMC
    gap_samples = b["gap"] + b["sg"] * z[:, 0]
    speed_samples = np.maximum(0., b["vlead"] + b["sv"] * (
        b["rho"] * z[:, 0] + np.sqrt(max(0., 1. - b["rho"] ** 2)) * z[:, 1]))
    sample_states = np.empty((len(actions), QMC_SAMPLES, 5), dtype=np.float64)
    sample_states[..., 0] = gap_samples
    sample_states[..., 1] = ve
    sample_states[..., 2] = speed_samples
    sample_states[..., 3] = pending
    sample_states[..., 4] = actions[:, None]
    conditional, outcomes = backup_risk(sample_states, return_details=True)
    risks = b["presence"] * conditional
    point = np.column_stack((np.full(len(actions), b["gap"]),
                             np.full(len(actions), ve), np.full(len(actions), max(0., b["vlead"])),
                             np.full(len(actions), pending), actions))
    point_margin = backup_outcomes(point[:, None, :])["minimum_gap_m"][:, 0]
    k = p if kind == "rss" else FALLBACK_RSS_K
    rss = point.copy()
    rss[:, 0] -= k * b["sg"]
    rss[:, 2] = max(0., b["vlead"] - k * b["sv"])
    rss_margin = backup_outcomes(rss[:, None, :])["minimum_gap_m"][:, 0]
    ttc = commitment_ttc(b["gap"], ve, max(0., b["vlead"]), pending, actions)
    fallback = not b["has_track"]
    if fallback:
        safe = rss_margin >= 0.
    elif kind == "risk":
        safe = risks <= p
    elif kind == "rss":
        safe = rss_margin >= 0.
    else:
        # Presence=0 is an explicit absence belief; missing track instead uses RSS.
        safe = ~((b["presence"] > 0.) & (ttc < p))
    if not b["valid"]:
        safe = np.zeros(len(actions), dtype=bool)
    return dict(candidate_actions=actions.copy(), safe=np.asarray(safe, dtype=bool),
                risk=risks, conditional_risk=conditional,
                minimum_gap_m=outcomes["minimum_gap_m"].min(axis=-1),
                point_margin_m=point_margin, rss_margin_m=rss_margin, ttc_s=ttc,
                presence=b["presence"], has_track=b["has_track"], belief_valid=b["valid"],
                fallback_to_rss=fallback, rss_k=k, kind=kind, param=p,
                qmc_samples=QMC_SAMPLES, semantics=SEMANTICS)


def evaluate_candidate(belief, ego_speed, pending_acc, candidate_acc,
                       kind="risk", param=.01):
    """JSON-friendly diagnostics for any selected action, including fallback -6."""
    details = evaluate_candidates(belief, ego_speed, pending_acc, [candidate_acc], kind, param)
    return {key: (value[0].item() if isinstance(value, np.ndarray) else value)
            for key, value in details.items()}


def nominal_acceleration(ego_speed, target_speed=22.):
    """Shared speed-tracking nominal, clipped to [-3,3] m/s^2."""
    if not np.isfinite(target_speed) or target_speed < 0:
        raise ValueError("Target speed must be finite and nonnegative")
    return float(np.clip((float(target_speed) - float(ego_speed)) * .8, -3., 3.))


def choose_action(belief, ego_speed, pending_acc, kind="risk", param=.01, target_speed=22.):
    """Largest safe ACTIONS member <= nominal; full braking if none qualifies."""
    details = choose_action_details(belief, ego_speed, pending_acc, kind, param, target_speed)
    return details["command"]


def choose_action_details(belief, ego_speed, pending_acc, kind="risk", param=.01, target_speed=22.):
    """Selected action plus all candidate and selected risk/margin diagnostics."""
    nominal = nominal_acceleration(ego_speed, target_speed)
    details = evaluate_candidates(belief, ego_speed, pending_acc, ACTIONS, kind, param)
    eligible = details["safe"] & (ACTIONS <= nominal + 1e-12)
    selected = int(np.flatnonzero(eligible)[-1]) if eligible.any() else 0
    selected_details = {key: (value[selected].item() if isinstance(value, np.ndarray) else value)
                        for key, value in details.items()}
    return dict(command=float(ACTIONS[selected]), nominal=nominal,
                fallback=not bool(eligible.any()), selected_index=selected,
                selected=selected_details, candidates=details)


def evaluate_batch(beliefs, ego_speeds, pending_accs, candidates=ACTIONS, kind="risk", param=.01):
    """Convenience batch; vectorize each state's candidate x quadrature matrix.

    ego_speeds/pending_accs/param may be scalars or one value per belief. Return
    one evaluate_candidates dictionary per belief. No random stream is created.
    """
    beliefs = list(beliefs)
    n = len(beliefs)
    speeds = np.broadcast_to(np.asarray(ego_speeds, dtype=float), (n,))
    pending = np.broadcast_to(np.asarray(pending_accs, dtype=float), (n,))
    parameters = np.broadcast_to(np.asarray(param, dtype=float), (n,))
    return [evaluate_candidates(b, speeds[i], pending[i], candidates, kind, parameters[i])
            for i, b in enumerate(beliefs)]


batch_evaluate = evaluate_batch
