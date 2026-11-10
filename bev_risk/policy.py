"""Deterministic offline stopping-support contract; not a collision probability."""
from __future__ import annotations
import math

ACTIONS = (-6., -3., 0., 1.5, 3.)


def nominal_action(ego_speed: float) -> float:
    command = max(-3., min(3., .8 * (22. - max(0., ego_speed))))
    return max(a for a in ACTIONS if a <= command)


def stopping_distance(ego_speed: float, action: float) -> float:
    """0.2 s coast, up to 0.2 s action, then full braking at 6 m/s²."""
    v = max(0., float(ego_speed))
    duration = min(.2, v / -action) if action < 0 else .2
    return .2*v + v*duration + .5*action*duration**2 + max(0., v+action*duration)**2/12.


def stress_shift(gap: float, speed: float, age: float, allowance: float | None):
    v = max(0., speed)
    h = max(0., age-allowance) if allowance is not None else 0.
    u = min(h, v/3.)
    return gap - (v*(h-u)+1.5*u*u), v-3.*u


def choose_action(gap: float | None, lead_speed: float | None, ego_speed: float,
                  age_s: float = 0., allowance_s: float | None = None, buffer_m: float = 0.) -> dict:
    nominal = nominal_action(ego_speed)
    if gap is None or lead_speed is None:
        return dict(action=-6., nominal_action=nominal, reason="no_estimated_lead",
                    corrected_gap_m=None, corrected_speed_mps=None, support_m=None, age_active=False, correction_nonzero=False)
    if not all(math.isfinite(x) for x in (gap, lead_speed, ego_speed, age_s, buffer_m)):
        raise ValueError("Nonfinite policy input")
    g, v = stress_shift(gap, lead_speed, age_s, allowance_s)
    admitted = [a for a in ACTIONS if a <= nominal and g > max(0., stopping_distance(ego_speed, a)-v*v/16.)+buffer_m]
    return dict(action=max(admitted) if admitted else -6., nominal_action=nominal,
                reason="admitted" if admitted else "full_brake_fallback",
                corrected_gap_m=g, corrected_speed_mps=v, support_m=g+v*v/16.,
                age_active=allowance_s is not None and age_s > allowance_s,
                correction_nonzero=abs(g-gap)>1e-9 or abs(v-max(0.,lead_speed))>1e-9)


def admission(candidate: dict, mode: str, ego_speed: float, past_speed: float,
              ego_front_offset_m: float = 2.5) -> tuple[bool, float]:
    score = candidate.get("camera_score", 0.) or 0.
    margin = (candidate["observed_rear_x_ego_m"]-ego_front_offset_m
              + max(0., past_speed)**2/16. - stopping_distance(ego_speed, nominal_action(ego_speed)))
    if mode == "L":
        accepted = True
    elif mode == "H":
        accepted = score >= .50
    elif mode == "S":
        accepted = score >= .20
    elif mode == "R":
        accepted = score >= .50 or (score >= .20 and margin <= 2.)
    else:
        raise ValueError(mode)
    return bool(accepted), float(margin)
