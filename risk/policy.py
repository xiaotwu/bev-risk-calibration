"""The frozen CV and fixed age-correction decisions for supplied observations."""
import math

import numpy as np

from . import control as C
from .tail_integral import integrated_risk

BRAKING_MPS2 = 3.0
FRESH_ALLOWANCE_S = 1.0


def state_belief(tr,presence,sigma_gap,sigma_v):
    return dict(mean_gap=float(tr['mean_gap']),mean_vlead=float(tr['mean_vlead']),
        presence=float(presence),sigma_gap=float(sigma_gap),sigma_v=float(sigma_v),
        covariance=tr['covariance'],has_track=bool(tr['feature'][14] or tr['presence']<.2),age_s=float(tr['age_s']))


def choose_integrated(belief,ego_speed,pending,*,epsilon):
    details=C.evaluate_candidates(belief,ego_speed,pending,kind='risk',param=epsilon)
    details['minimum_gap_semantics']='128_QMC_reference_only_not_used_for_risk_admission'
    if details['belief_valid']:
        integral=integrated_risk(belief,ego_speed,pending,C.ACTIONS)
        details.update(risk=integral['risk'],conditional_risk=integral['conditional_risk'],
            integrator='split_1d_conditional_gaussian',numerical_error_estimate=integral['numerical_error_estimate'],
            error_estimate_is_certified=False)
        if not details['fallback_to_rss']:details['safe']=integral['risk']<=epsilon
    else:details['integrator']='invalid_belief_full_braking'
    nominal=C.nominal_acceleration(ego_speed)
    eligible=details['safe']&(C.ACTIONS<=nominal+1e-12)
    selected=int(np.flatnonzero(eligible)[-1]) if eligible.any() else 0
    d={k:(v[selected].item() if isinstance(v,np.ndarray) else v) for k,v in details.items()}
    return dict(command=float(C.ACTIONS[selected]),nominal=nominal,selected_index=selected,
        fallback=not bool(eligible.any()),selected=d,candidates=details)


def stress_shift(raw_velocity, last_positive_age):
    """Subtract the deficit versus CV, with a stop and no reversal."""
    horizon = max(0., last_positive_age - FRESH_ALLOWANCE_S)
    speed = max(0., raw_velocity)
    active = min(horizon, speed / BRAKING_MPS2)
    velocity_deficit = BRAKING_MPS2 * active
    gap_deficit = speed * (horizon - active) + .5 * BRAKING_MPS2 * active**2
    return horizon, gap_deficit, velocity_deficit


class MotionPolicy:
    def __init__(self, policy, fits):
        self.tracker = policy['tracker']
        self.calibration = fits[self.tracker]
        self.prior_calibration = fits['cv_current']
        self.epsilon = float(policy['param'])
        self.use_stress = policy.get('age_stress', False)

    def step(self, tracker, ego_speed, pending, time):
        positive = bool(tracker['feature'][14])
        usable = positive or tracker['presence'] < .2
        actual = tracker if positive else tracker['legacy_reference']
        calibration = self.calibration if positive else self.prior_calibration
        age = float(np.float32(actual['feature'][2]))
        age_bin = int(np.searchsorted([.5, 1., 2.], max(0., age), side='right'))
        factor = calibration['age_bins'][str(age_bin)]['gaussian_equivalent_factor']
        belief = state_belief(actual, actual['presence'], max(.05, actual['base_sigma_gap'] * factor),
                             max(.1, actual['base_sigma_v'] * factor))
        stamp = tracker['tracker_diagnostics']['last_detection_acquisition_s']
        last_positive_age = None
        horizon = gap_deficit = velocity_deficit = 0.
        if positive:
            if stamp is None or not math.isfinite(stamp) or stamp > time + 1e-6:
                raise ValueError('Invalid delivered positive acquisition time')
            last_positive_age = max(0., time - stamp)
            if self.use_stress:
                horizon, gap_deficit, velocity_deficit = stress_shift(belief['mean_vlead'], last_positive_age)
        original_gap = belief['mean_gap']
        original_velocity = belief['mean_vlead']
        belief['mean_gap'] -= gap_deficit
        belief['mean_vlead'] -= velocity_deficit
        belief.update(tracker=self.tracker, physical_scale_age_used_s=age)
        decision = choose_integrated(belief, ego_speed, pending, epsilon=self.epsilon if usable else .001)
        decision.update(common_unseen_prior=not positive, tracked_threshold_used=usable, age_stress=dict(
            enabled=self.use_stress, positive_acquisition_age_s=last_positive_age,
            stress_duration_s=horizon, gap_deficit_m=gap_deficit, velocity_deficit_mps=velocity_deficit,
            original_gap_m=original_gap, original_velocity_mps=original_velocity))
        return dict(command=decision['command'], belief=belief, diagnostics=decision)
