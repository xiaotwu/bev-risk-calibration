"""Deterministic 1-D Gaussian integration for the preserved backup contract.

This computes MODEL risk.  Accurate integration cannot validate the Gaussian
belief, its presence probability, empirical calibration, or real safety.

The ego's two .2-second commands are >= -6 and its final brake is -6; the
leader brakes at -8.  Until the leader stops, relative closing speed is
nondecreasing, including ego stop/restart events.  Afterwards closing speed
is nonnegative.  Consequently the largest ego-minus-leader displacement is
the maximum of zero and the final stopping-distance difference.  Contact is
exactly G <= max(0, E - max(0, Vraw)**2/16).

Condition on Z=(Vraw-mu_v)/sigma_v.  G|Z has mean mu_g+rho*sigma_g*Z
and variance sigma_g**2*(1-rho**2).  The clipped-speed atom is retained by
integrating the entire Z<=-mu_v/sigma_v region WITH its G correlation.

Polynomial crossing splits resolve both clipping and narrow conditional-CDF
transitions.  The fast rule uses deterministic Gauss-Legendre quadrature.
Its comparison to a lower-order rule is an error ESTIMATE, not a certified
enclosure.  Explicit omitted normal-tail mass is bounded separately.
"""
from __future__ import annotations

import functools
import math
from collections.abc import Mapping
import numpy as np
from scipy.special import ndtr

DT = .2
EGO_BRAKE = 6.
LEADER_BRAKE = 8.
Z_LIMIT = 10.
TAIL_MASS_BOUND = float(2 * ndtr(-Z_LIMIT))
SEMANTICS = 'pending_0.2s_then_candidate_0.2s_then_ego_brake6_leader_brake8'
_NORMALIZER = 1 / math.sqrt(2 * math.pi)


def _displacement(v, a, dt):
    h = min(dt, v / -a) if a < 0 else dt
    return v * h + .5 * a * h*h


def ego_stopping_distance(ego_speed, pending_acc, candidate_acc):
    ve, pending, candidate = map(float, (ego_speed, pending_acc, candidate_acc))
    if (not all(map(math.isfinite, (ve, pending, candidate))) or ve < 0
            or not -6 <= pending <= 3 or not -6 <= candidate <= 3):
        raise ValueError('Finite nonnegative speed and commands in [-6,3] required')
    e1 = _displacement(ve, pending, DT)
    v1 = max(0., ve + pending * DT)
    e2 = _displacement(v1, candidate, DT)
    v2 = max(0., v1 + candidate * DT)
    return e1 + e2 + v2*v2 / (2 * EGO_BRAKE)


def required_clearance(leader_speed, ego_speed, pending_acc, candidate_acc):
    speed = np.asarray(leader_speed, dtype=float)
    if not np.isfinite(speed).all() or (speed < 0).any():
        raise ValueError('Actual leader speeds must be finite and nonnegative')
    return np.maximum(0., ego_stopping_distance(ego_speed, pending_acc, candidate_acc)
                      - speed*speed / (2 * LEADER_BRAKE))


def _mass(lo, hi):
    """Stable standard-normal probability of an interval."""
    return ndtr(-lo) - ndtr(-hi) if lo >= 0 else ndtr(hi) - ndtr(lo)


def _roots(a, b, c):
    if a == 0:
        return [] if b == 0 else [-c / b]
    disc = b*b - 4*a*c
    if disc < 0:
        return []
    d = math.sqrt(disc)
    q = -.5 * (b + math.copysign(d, b))
    if q == 0:
        return [-b/(2*a)]
    return [q/a, c/q]


def _pieces(E, mg, mv, sv, A):
    z0 = -mv / sv
    z1 = (math.sqrt(2 * LEADER_BRAKE * E) - mv) / sv
    return [(float('-inf'), z0, (0., -A, E-mg)),
            (z0, z1, (-sv*sv/16., -mv*sv/8.-A, E-mg-mv*mv/16.)),
            (z1, float('inf'), (0., -A, -mg))]


def _polynomial(coef, z):
    return (coef[0] * z + coef[1]) * z + coef[2]


def _split_intervals(pieces, sd, limit):
    result = []
    for raw_lo, raw_hi, coef in pieces:
        lo, hi = max(raw_lo, -limit), min(raw_hi, limit)
        if not lo < hi:
            continue
        boundaries = [lo, hi]
        # Fixed normal-density knots avoid broad intervals spanning its peak.
        boundaries += [x for x in (-8., -5., -2., 0., 2., 5., 8.) if lo < x < hi]
        a, b, c = coef
        if a:
            vertex = -b / (2*a)
            if lo < vertex < hi:
                boundaries.append(vertex)
        # Solve in unscaled physical units, avoiding ill conditioning for tiny sd.
        for level in (-10., -6., -3., 0., 3., 6., 10.):
            boundaries += [x for x in _roots(a, b, c-level*sd) if lo < x < hi]
        boundaries = sorted(set(boundaries))
        result.extend((a0, b0, coef) for a0, b0 in zip(boundaries[:-1], boundaries[1:])
                      if a0 < b0)
    return result


@functools.lru_cache(maxsize=8)
def _legendre(order):
    if order < 8 or order > 256:
        raise ValueError('Gauss-Legendre order must be in [8,256]')
    x, w = np.polynomial.legendre.leggauss(order)
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def _integrate_rule(intervals, sd, order):
    if not intervals:
        return 0.
    arr = np.array([[a, b, *c] for a, b, c in intervals], dtype=float)
    midpoint = .5 * (arr[:, 0] + arr[:, 1])
    halfwidth = .5 * (arr[:, 1] - arr[:, 0])
    x, weights = _legendre(order)
    z = midpoint[:, None] + halfwidth[:, None] * x
    h = ((arr[:, 2, None] * z + arr[:, 3, None]) * z + arr[:, 4, None]) / sd
    values = ndtr(h) * np.exp(-.5*z*z) * _NORMALIZER
    return float(np.sum(halfwidth * (values @ weights)))


def _rank_one_probability(pieces):
    """Exact polynomial-event integration for rho=+/-1 (up to roundoff)."""
    total = 0.
    for lo, hi, coef in pieces:
        if not lo < hi:
            continue
        a, b, c = coef
        roots = sorted(set(x for x in _roots(a, b, c) if lo < x < hi))
        bounds = [lo, *roots, hi]
        for left, right in zip(bounds[:-1], bounds[1:]):
            if left == float('-inf'):
                mid = right-1 if math.isfinite(right) else 0.
            elif right == float('inf'):
                mid = left+1
            else:
                mid = .5*(left+right)
            if _polynomial(coef, mid) >= 0:
                total += _mass(left, right)
    return float(np.clip(total, 0., 1.))


def conditional_contact_probability(mean_gap, mean_vlead, sigma_gap, sigma_v,
                                    rho, ego_speed, pending_acc, candidate_acc,
                                    *, order=32, return_details=False):
    """Probability of backup contact conditional on existence, under the model.

    ``mean_vlead`` is the RAW Gaussian mean before physical support clipping.
    A details dict reports a numerical error estimate (NOT a safety bound).
    All motion inputs use metres, seconds, and the original fixed contract.
    """
    mg, mv, sg, sv, rho = map(float, (mean_gap, mean_vlead, sigma_gap, sigma_v, rho))
    if (not all(map(math.isfinite, (mg, mv, sg, sv, rho))) or min(sg, sv) < 0
            or abs(rho) > 1):
        raise ValueError('Finite Gaussian parameters, nonnegative scales, rho in [-1,1] required')
    E = ego_stopping_distance(ego_speed, pending_acc, candidate_acc)
    method, intervals, low, estimate = 'closed_form_degenerate', [], None, 0.
    if sg == 0:
        if mg <= 0:
            prob = 1.
        elif mg > E:
            prob = 0.
        elif sv == 0:
            prob = float(mg <= max(0., E-max(0., mv)**2/16.))
        else:
            prob = float(ndtr((math.sqrt(16.*(E-mg))-mv)/sv))
    elif sv == 0:
        D = max(0., E-max(0., mv)**2/16.)
        prob = float(ndtr((D-mg)/sg))
    else:
        pieces = _pieces(E, mg, mv, sv, rho*sg)
        sd = sg * math.sqrt(max(0., 1-rho*rho))
        if sd == 0:
            method = 'closed_form_rank_one_polynomial_event'
            prob = _rank_one_probability(pieces)
        else:
            method = 'split_conditional_gaussian_gauss_legendre'
            intervals = _split_intervals(pieces, sd, Z_LIMIT)
            prob = _integrate_rule(intervals, sd, int(order))
            low = _integrate_rule(intervals, sd, max(8, int(order)//2))
            estimate = abs(prob-low) + TAIL_MASS_BOUND + 64*np.finfo(float).eps
    prob = float(np.clip(prob, 0., 1.))
    if not return_details:
        return prob
    return dict(probability=prob, numerical_error_estimate=estimate,
                error_estimate_is_certified=False, omitted_normal_tail_mass_bound=TAIL_MASS_BOUND,
                lower_order_probability=low, order=int(order), intervals=len(intervals),
                method=method, ego_stopping_distance_m=E, semantics=SEMANTICS,
                guarantee='Gaussian-model integration only; no empirical calibration or safety guarantee')


def adaptive_reference(mean_gap, mean_vlead, sigma_gap, sigma_v, rho,
                       ego_speed, pending_acc, candidate_acc, *, epsabs=2e-12):
    """Independent high-accuracy adaptive-quad reference, not the online rule."""
    from scipy.integrate import quad
    mg, mv, sg, sv, rho = map(float, (mean_gap, mean_vlead, sigma_gap, sigma_v, rho))
    if sg == 0 or sv == 0 or abs(rho) == 1:
        p = conditional_contact_probability(mg, mv, sg, sv, rho,
                ego_speed, pending_acc, candidate_acc)
        return dict(probability=p, estimated_quad_error=0., method='exact_degenerate')
    E = ego_stopping_distance(ego_speed, pending_acc, candidate_acc)
    sd = sg*math.sqrt(1-rho*rho)
    intervals = _split_intervals(_pieces(E, mg, mv, sv, rho*sg), sd, 12.)
    value, error = 0., 0.
    for lo, hi, coef in intervals:
        fn = lambda z: float(ndtr(_polynomial(coef,z)/sd))*math.exp(-.5*z*z)*_NORMALIZER
        val, err = quad(fn, lo, hi, epsabs=epsabs/max(1,len(intervals)), epsrel=2e-11, limit=100)
        value += val
        error += err
    return dict(probability=float(np.clip(value,0.,1.)), estimated_quad_error=error+2*ndtr(-12.),
                method='scipy_adaptive_quad_split_conditional_gaussian')


def _get(belief, name, default=None):
    return belief.get(name, default) if isinstance(belief, Mapping) else getattr(belief,name,default)


def integrated_risk(belief, ego_speed, pending_acc,
                    candidates=(-6., -3., 0., 1.5, 3.), *, order=32):
    """Candidate model risks; this function does not choose a command/fallback.

    Optional covariance supplies correlation only.  Marginal sigmas always
    come from the supplied belief, exactly as in the original controller.
    """
    mg, mv, sg, sv = [float(_get(belief,k,float('nan'))) for k in
                     ('mean_gap','mean_vlead','sigma_gap','sigma_v')]
    presence = float(_get(belief,'presence',1.))
    if not math.isfinite(presence) or not 0 <= presence <= 1:
        raise ValueError('Presence must be a finite probability')
    rho = _get(belief,'rho',_get(belief,'correlation',None))
    cov = _get(belief,'cov',_get(belief,'covariance',None))
    if rho is None and cov is not None:
        c = np.asarray(cov,dtype=float)
        if (c.shape != (2,2) or not np.isfinite(c).all() or np.any(np.diag(c)<0)
                or not np.allclose(c,c.T,rtol=1e-7,atol=1e-10)):
            raise ValueError('Covariance must be finite symmetric 2x2 with nonnegative diagonal')
        denom = math.sqrt(c[0,0]*c[1,1])
        if abs(c[0,1]) > denom+1e-8:
            raise ValueError('Covariance must be positive semidefinite')
        rho = c[0,1]/denom if denom else 0.
    rho = 0. if rho is None else float(rho)
    if math.isfinite(rho) and abs(rho) <= 1+1e-8:
        rho = float(np.clip(rho,-1.,1.))
    actions = np.asarray(candidates,dtype=float)
    if actions.ndim != 1 or not len(actions):
        raise ValueError('Nonempty one-dimensional candidates required')
    details = [conditional_contact_probability(mg,mv,sg,sv,rho,ego_speed,pending_acc,a,
                    order=order,return_details=True) for a in actions]
    conditional = np.array([d['probability'] for d in details])
    estimates = presence*np.array([d['numerical_error_estimate'] for d in details])
    return dict(candidate_actions=actions,conditional_risk=conditional,risk=presence*conditional,
                numerical_error_estimate=estimates,error_estimate_is_certified=False,
                presence=presence,rho=rho,details=details,semantics=SEMANTICS)
