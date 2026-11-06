"""Namespaces for the committed-pose retest and independent cadence stress."""
import numpy as np

STUDY_ID = 'bev_age_risk_v5'
ROOT_ENTROPY = (20261105, 294818235, 1096692577, 3456172909)
ROLE = dict(pilot=510, pilot_retest=511, pilot_commit=512, calibration=520, development=530, diagnostic=540, stress=550)
COUNTS = dict(pilot=16, pilot_retest=16, pilot_commit=16, calibration=256, development=96, diagnostic=512, stress=96)
DT = .1
HORIZON_S = HORIZON = 8.
CONTROL_EVERY = 2
BEV_EVERY = 10
EPSILONS = (.001, .01, .05)


def rng_for(role, index, stream):
    if role not in ROLE or type(index) is not int or not 0 <= index < COUNTS[role]:
        raise ValueError('Unknown v5 role or scene index')
    if type(stream) is not int or stream not in (1, 2, 3, 4, 5, 6):
        raise ValueError('Unknown v5 random stream')
    seed = np.random.SeedSequence(ROOT_ENTROPY, spawn_key=(ROLE[role], index, stream))
    return np.random.default_rng(seed)


def scenario(role, index):
    r = rng_for(role, index, 1)
    ve = float(r.uniform(10, 22))
    present = bool(r.random() < .8)
    physical = dict(initial_gap=float(r.uniform(12, 38)), ve=ve,
        vl=max(0., ve - float(r.uniform(0, 5))),
        lead_onset=float(r.uniform(.5, 3.)) if r.random() < .7 else 100.,
        b=float(r.uniform(3, 8)), lead_present=present,
        seed=int(rng_for(role, index, 2).integers(0, 2147483647)))
    channel = rng_for(role, index, 3)
    spec = dict(condition=str(channel.choice(['clean', 'blackout', 'stale'], p=[.35, .35, .30])),
        start=float(channel.uniform(.5, 2.5)), duration=float(channel.uniform(.6, 2.5)),
        delay=float(channel.uniform(.2, 1.5)))
    spec['keep'] = (channel.random(81) >= .03).tolist()
    return physical, spec


def delivered(scan, now, channel):
    tick = int(round(scan.episode_time_s / DT))
    if tick < 0 or tick >= len(channel['keep']) or not channel['keep'][tick]:
        return False
    t = scan.episode_time_s
    affected = channel['start'] <= t < channel['start'] + channel['duration']
    if affected and channel['condition'] == 'blackout':
        return False
    delay = channel['delay'] if affected and channel['condition'] == 'stale' else 0.
    return t + delay <= now + 1e-7


def candidate_period(role, index):
    return int(rng_for(role, index, 5).integers(0, 2))


def pilot_order(index):
    return rng_for('pilot_commit', index, 6).permutation(4).tolist()


def bev_every(role):
    return 20 if role == 'stress' else BEV_EVERY
