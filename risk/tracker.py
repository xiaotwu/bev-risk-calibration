"""Constant-velocity Kalman tracking with a causal packet history.

This is the CV-only portion of the simulation tracker. An observation is a
delivered detector packet in the current ego frame or None when unavailable.
"""
from __future__ import annotations

import numpy as np

ROI_DIM = 384
CORRIDOR_DIM = 15
VIRTUAL_R = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
VERSION = 'causal_motion_v4_1'


def _vector(value, size, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite with shape({size},)')
    return a


def _half_extent(front):
    box = np.asarray(front.get('box_lidar', []), dtype=np.float64)
    if len(box) >= 7 and np.isfinite(box[:7]).all() and np.all(box[3:6] > 0):
        direction = np.array([np.cos(box[6]), np.sin(box[6])])
        sideways = np.array([-np.sin(box[6]), np.cos(box[6])])
        return float(.5*(abs(VIRTUAL_R[0,:2]@direction)*box[3]
                          + abs(VIRTUAL_R[0,:2]@sideways)*box[4]))
    # Explicit detector-metadata fallback, never a true actor dimension.
    return 2.25


class BeliefTracker:
    """Shared causal 2-state KF and an explicit Bayesian existence heuristic.

    No observation means prediction only. A real negative detector packet is
    distinct from no packet. frame/acquisition_time identity prevents an old raw
    packet from updating twice; the same dict object is additionally rejected.
    Negative packets update the existence heuristic, not the continuous KF.
    """
    def __init__(self, *, process_accel_std=2.5, position_std=.4,
                 velocity_std=1., prior_presence=.8,
                 detection_probability=.85, false_front_probability=.05,
                 existence_switch_rate=.005, reset_position_m=12.):
        if not 0 < prior_presence < 1: raise ValueError('prior_presence must be in(0,1)')
        if not 0 < false_front_probability < detection_probability < 1:
            raise ValueError('Require 0<p_false<p_detect<1')
        self.config = dict(process_accel_std=float(process_accel_std),
            position_std=float(position_std), velocity_std=float(velocity_std),
            prior_presence=float(prior_presence), detection_probability=float(detection_probability),
            false_front_probability=float(false_front_probability),
            existence_switch_rate=float(existence_switch_rate), reset_position_m=float(reset_position_m))
        self.reset()

    def reset(self):
        self.state = None; self.P = None; self.time = None
        self.existence = self.config['prior_presence']; self.half_extent = 2.25
        self.ever_detected = False; self.last_detection_acquisition = None
        self.last_key = None; self.last_object = None
        self.last_roi = np.zeros(ROI_DIM); self.last_corridor = np.zeros(CORRIDOR_DIM)
        self.last_innovation = np.zeros(2); self.initial_time = None

    def update(self, observation, ego_x, ego_speed, pending_acc, time,
               ego_front_offset=2.0):
        values = np.asarray([ego_x, ego_speed, pending_acc, time, ego_front_offset], float)
        if not np.isfinite(values).all() or ego_speed < 0:
            raise ValueError('Known ego state/time must be finite; speed nonnegative')
        now = float(time)
        if self.time is not None and now <= self.time:
            raise ValueError('Tracker time must increase strictly; reset at episode boundaries')
        if self.state is None:
            self.state = np.array([ego_x+ego_front_offset+self.half_extent+25., ego_speed], float)
            self.P = np.diag([12.**2, 8.**2]); self.initial_time = now
        dt = 0. if self.time is None else now-self.time
        if dt:
            F = np.array([[1., dt], [0., 1.]])
            G = np.array([.5*dt*dt, dt])
            self.state = F@self.state
            self.P = F@self.P@F.T + self.config['process_accel_std']**2*np.outer(G,G)
            switch = min(.25, self.config['existence_switch_rate']*dt)
            self.existence += switch*(1.-2.*self.existence)
        packet = observation is not None
        unique = False; used = False; raw_detected = False; score = 0.
        innovation = np.zeros(2); previous_innovation = self.last_innovation.copy()
        raw_age = 0.; acquisition = now
        if packet:
            if not isinstance(observation, dict): raise ValueError('observation must be dict or None')
            raw_age = float(observation.get('sensor_age_s', 0.))
            if not np.isfinite(raw_age) or raw_age < 0: raise ValueError('sensor_age_s must be nonnegative')
            acquisition = float(observation.get('acquisition_time', now-raw_age))
            if not np.isfinite(acquisition) or acquisition > now+1e-6:
                raise ValueError('Acquisition time cannot be in the future')
            # Prefer raw sensor frame. The current ego-projected BEV can change
            # while the old raw packet is unchanged; do not count it twice.
            if observation.get('frame') is not None:
                key = ('frame', str(observation['frame']))
            elif observation.get('packet_id') is not None:
                key = ('packet', str(observation['packet_id']))
            else:
                key = ('acquisition', acquisition)
            has_source_identity = any(observation.get(k) is not None for k in ['frame','packet_id','acquisition_time'])
            unique = key != self.last_key and (has_source_identity or observation is not self.last_object)
            front = observation.get('front'); raw_detected = front is not None
            if unique:
                self.last_key = key; self.last_object = observation
                self.last_corridor = _vector(observation.get('corridor', np.zeros(15)),15,'corridor')
                if front is not None:
                    center = _vector(front.get('center', front.get('center_ego')),3,'front.center')
                    velocity = _vector(front.get('velocity', front.get('velocity_ego')),3,'front.velocity')
                    score = float(front.get('score', 0.))
                    if not np.isfinite(score) or not 0 <= score <= 1: raise ValueError('score must be in[0,1]')
                    z = np.array([ego_x+center[0], velocity[0]])
                    self.half_extent = _half_extent(front)
                    innovation = z-self.state
                    # Treat old dynamically uncorrected objects as current
                    # measurements with explicit age-inflated noise.
                    pos_variance = self.config['position_std']**2 + (2.*raw_age)**2 + .25*(8.*raw_age*raw_age)**2
                    vel_variance = self.config['velocity_std']**2 + (8.*raw_age)**2
                    R = np.diag([pos_variance, vel_variance])
                    if not self.ever_detected or abs(innovation[0]) > self.config['reset_position_m']:
                        self.state = z.copy(); self.P = R.copy()
                    else:
                        K = np.linalg.solve((self.P+R).T,self.P.T).T
                        self.state = self.state+K@innovation
                        I_K = np.eye(2)-K
                        self.P = I_K@self.P@I_K.T+K@R@K.T
                    self.ever_detected = True; used = True
                    self.last_detection_acquisition = acquisition
                    self.last_roi = _vector(observation.get('roi',np.zeros(384)),384,'roi')
                    self.last_innovation = innovation.copy()
                    yes = self.config['detection_probability']; no = self.config['false_front_probability']
                else:
                    yes = 1.-self.config['detection_probability']; no = 1.-self.config['false_front_probability']
                p=self.existence; self.existence=float(np.clip(p*yes/(p*yes+(1-p)*no),1e-4,1-1e-4))
        self.time = now
        age = (now-self.last_detection_acquisition if self.last_detection_acquisition is not None
               else now-self.initial_time)
        age = max(float(age),raw_age)
        gap = float(self.state[0]-ego_x-self.half_extent-ego_front_offset)
        vlead = float(self.state[1])
        # Expose untruncated mean; the scenario sampler, not the estimator,
        # must handle the physical nonnegative-speed support explicitly.
        sigma = np.sqrt(np.maximum(np.diag(self.P),[.05**2,.1**2]))
        feature = np.array([float(used),float(packet),age,score,gap,vlead,ego_speed,pending_acc,
            innovation[0] if used else 0.,innovation[1] if used else 0.,
            previous_innovation[0],previous_innovation[1],sigma[0],sigma[1],float(self.ever_detected)],np.float32)
        return dict(feature=feature,features=feature,roi=self.last_roi.astype(np.float32).copy(),
            corridor=self.last_corridor.astype(np.float32).copy(),mean_gap=gap,mean_vlead=vlead,
            base_sigma_gap=float(sigma[0]),base_sigma_v=float(sigma[1]),presence=float(self.existence),
            covariance=self.P.copy(),detected=bool(used),raw_front_detected=bool(raw_detected),
            packet_received=bool(packet),measurement_used=bool(used),unique_packet=bool(unique),
            age_s=age,raw_sensor_age_s=raw_age,half_extent_ego_x=float(self.half_extent),
            world_center_mean=float(self.state[0]),time=now)


def _legacy_reference(row):
    """Minimal, nonrecursive snapshot for the common no-detection fallback.

    Capture this BEFORE replacing the new estimator's mean, covariance or
    feature vector. The policy uses one shared CV calibration on this reference
    until observable packet history makes its own tracker usable.
    """
    fields=('feature','mean_gap','mean_vlead','base_sigma_gap','base_sigma_v',
            'covariance','presence','age_s')
    return {k:(row[k].copy() if isinstance(row[k],np.ndarray) else row[k])
            for k in fields}


class LegacyCVTracker:
    """Strict legacy mean/P/presence, with additive trace diagnostics only."""
    def __init__(self):
        self._tracker = BeliefTracker()
        self.kind='cv_current'
    def reset(self):self._tracker.reset()
    def update(self,*args,**kwargs):
        row=self._tracker.update(*args,**kwargs)
        row['legacy_reference']=_legacy_reference(row)
        state=self._tracker.state.copy();p=self._tracker.P.copy()
        row['state3']=np.r_[state,0.]
        row['P3']=np.pad(p,((0,1),(0,1)))
        row['estimated_acceleration_mps2']=0.
        row['tracker_diagnostics']=dict(kind=self.kind,version=VERSION,
            measurement_reference='legacy current-coordinate approximation',
            acquisition_anchor_is_exact=False,
            last_detection_acquisition_s=self._tracker.last_detection_acquisition,
            covariance_semantics='legacy two-state KF covariance')
        return row
