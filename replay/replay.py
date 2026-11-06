"""Outcome-blind matching and paired trajectory-only NGSIM replay."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
COLUMNS = ['vehicle_id', 'global_time', 'preceding', 'lane_id', 'local_y',
           'v_length', 'v_class', 'v_vel', 'frame_id']
KEY = ['vehicle_id', 'global_time']
AGES = (1, 2, 3)


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for data in iter(lambda: stream.read(1 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def ngsim_joins(block, data_root):
    path = (data_root / block['path']).resolve(strict=True)
    if not path.is_relative_to(data_root / 'raw_recordings'):
        raise ValueError(f"Recording file is outside the input directory: {block['id']}")
    if sha(path) != block['sha256']:
        raise ValueError(f"Input digest mismatch for {block['id']}")
    source = pd.read_csv(path, usecols=COLUMNS, compression='infer')
    if len(source) != block['rows'] or not (
            source.global_time.ge(block['start_ms']) & source.global_time.lt(block['end_ms'])).all():
        raise ValueError(f"Input count or time boundary mismatch for {block['id']}")
    observed = source.drop_duplicates()
    unique = observed.drop_duplicates(KEY, keep=False)
    clean = unique.dropna(subset=COLUMNS)
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError('Nonfinite trajectory fields')
    vehicles = clean[clean.v_class.eq(2) & clean.lane_id.between(1, 5)
                     & clean.v_length.mul(.3048).between(2., 8.)].copy()
    for key in ['local_y', 'v_vel', 'v_length']:
        vehicles[key] *= .3048
    follower = vehicles.loc[vehicles.preceding.gt(0) & vehicles.preceding.ne(vehicles.vehicle_id),
                            KEY + ['preceding', 'lane_id', 'frame_id', 'local_y']]
    leader = vehicles[KEY + ['lane_id', 'frame_id', 'local_y', 'v_length', 'v_vel']].rename(
        columns=dict(vehicle_id='preceding', local_y='lead_y', v_length='lead_length', v_vel='published_v'))
    pair = follower.merge(leader, on=['preceding', 'global_time', 'lane_id', 'frame_id'], validate='many_to_one')
    pair['gap'] = pair.lead_y - pair.lead_length - pair.local_y
    positive = pair[pair.gap.gt(0)].sort_values(KEY, ignore_index=True)
    audit = dict(raw_rows=len(source), identical_duplicate_rows=len(source)-len(observed),
                 conflicting_rows=len(observed)-len(unique), incomplete_rows=len(unique)-len(clean),
                 passenger_lane_rows=len(vehicles), matched_same_lane_rows=len(pair),
                 nonpositive_gap_rows=len(pair)-len(positive))
    return positive, audit


def eligibility(frame):
    run = (frame.vehicle_id.ne(frame.vehicle_id.shift()) |
           frame.global_time.diff().ne(100) |
           frame.frame_id.diff().ne(1) |
           frame.preceding.ne(frame.preceding.shift()) |
           frame.lane_id.ne(frame.lane_id.shift()) |
           frame.lead_length.ne(frame.lead_length.shift())).cumsum().to_numpy()
    time = frame.global_time.to_numpy(dtype='int64')
    idx = np.flatnonzero((time % 500 == 0) & frame.gap.between(2., 100.).to_numpy())
    idx = idx[(idx >= 6) & (idx + 33 < len(frame))]
    idx = idx[(run[idx-6] == run[idx]) & (run[idx+33] == run[idx])]
    position = frame.lead_y.to_numpy()
    pre = (position[idx] - position[idx-6]) / .6
    early = (position[idx+6] - position[idx]) / .6
    plausible = (pre >= 0) & (pre <= 50) & (early >= 0) & (early <= 50)
    idx, pre, early = idx[plausible], pre[plausible], early[plausible]
    anchors = frame.iloc[idx][KEY + ['preceding', 'lane_id', 'gap']].copy()
    anchors['row'] = idx
    anchors['speed'] = pre
    anchors['early_change'] = early - pre
    ordered = anchors.sort_values(['global_time', 'vehicle_id', 'preceding'])
    braking = ordered[ordered.early_change.le(-.75)]
    stable = ordered[ordered.early_change.abs().le(.25)]
    return braking, stable, len(ordered)


def match(braking, stable):
    stable_lanes = {lane: sub for lane, sub in stable.groupby('lane_id')}
    used = set()
    matched = []
    for brake in braking.itertuples(index=False):
        vehicles = {brake.vehicle_id, brake.preceding}
        if vehicles & used or brake.lane_id not in stable_lanes:
            continue
        peers = stable_lanes[brake.lane_id]
        ds = peers.speed.sub(brake.speed).abs()
        dg = peers.gap.sub(brake.gap).abs()
        possible = ((ds <= 3.) & (dg <= 10.) &
                    ~peers.vehicle_id.isin(used | vehicles) & ~peers.preceding.isin(used | vehicles))
        if not possible.any():
            continue
        choice = (ds[possible]/3. + dg[possible]/10.).idxmin()
        peer = peers.loc[choice]
        matched.append((brake.row, int(peer.row)))
        used.update(vehicles | {int(peer.vehicle_id), int(peer.preceding)})
    return np.asarray(matched, dtype='int64').reshape(-1, 2)


def score(frame, pairs, block_id, site):
    rows = pairs.reshape(-1)
    pos = frame.lead_y.to_numpy()
    ego = frame.local_y.to_numpy()
    raw_speed = frame.published_v.to_numpy()
    causal_speed = (pos[rows]-pos[rows-6])/.6
    out = []
    for age in AGES:
        future = rows + age * 10
        if not np.all(frame.global_time.to_numpy()[future] - frame.global_time.to_numpy()[rows] == age*1000):
            raise AssertionError('Scoring crosses missing acquisition')
        advance = pos[future] - pos[rows]
        anchor_gap = frame.gap.to_numpy()[rows]
        actual_gap = frame.gap.to_numpy()[future]
        if not np.allclose(actual_gap, anchor_gap + advance - (ego[future]-ego[rows])):
            raise AssertionError('Unexpected geometry or changing vehicle length')
        for method, v0, v1 in (
                ('position', causal_speed, (pos[future+3]-pos[future-3])/.6),
                ('published', raw_speed[rows], raw_speed[future])):
            position_error = v0*age - advance
            stopping_error = (np.maximum(v0, 0.)**2 - np.maximum(v1, 0.)**2)/16.
            cv = position_error + stopping_error
            direct = (anchor_gap + v0*age - (ego[future]-ego[rows]) +
                      np.maximum(v0, 0.)**2/16. - actual_gap - np.maximum(v1, 0.)**2/16.)
            if not np.allclose(cv, direct, atol=1e-9, rtol=0):
                raise AssertionError('B decomposition mismatch')
            horizon = max(0., age-1.)
            speed = np.maximum(v0, 0.)
            active = np.minimum(horizon, speed/3.)
            distance_shift = speed*(horizon-active) + 1.5*active**2
            stressed_v = np.maximum(v0-3.*active, 0.)
            stress = cv - distance_shift + (stressed_v**2-speed**2)/16.
            if np.any(stress > cv + 1e-10) or (age == 1 and not np.allclose(stress, cv, atol=1e-10, rtol=0)):
                raise AssertionError('R4 stress monotonicity/one-second parity failed')
            part = frame.iloc[rows][KEY + ['preceding', 'lane_id', 'frame_id', 'gap']].reset_index(drop=True)
            out.append(part.assign(block=block_id, site=site, age_s=age, velocity_source=method,
                                   match_id=np.repeat(np.arange(len(pairs)), 2),
                                   stratum=np.tile(['braking', 'stable'], len(pairs)),
                                   anchor_speed=causal_speed, scored_anchor_speed=v0, reference_speed=v1,
                                   leader_advance=advance, position_error=position_error,
                                   stopping_error=stopping_error, cv=cv, stress=stress))
    return pd.concat(out, ignore_index=True)


def metrics(group):
    cv = group.cv.to_numpy()
    stress = group.stress.to_numpy()
    return dict(n=len(group), cv_mean=float(cv.mean()), cv_median=float(np.median(cv)),
                cv_positive_rate=float((cv > 0).mean()),
                position_error_mean=float(group.position_error.mean()),
                stopping_error_mean=float(group.stopping_error.mean()),
                cv_positive_mean=float(np.maximum(cv, 0.).mean()),
                cv_negative_mean=float(np.maximum(-cv, 0.).mean()),
                stress_positive_mean=float(np.maximum(stress, 0.).mean()),
                stress_negative_mean=float(np.maximum(-stress, 0.).mean()),
                cv_mae=float(np.abs(cv).mean()), stress_mae=float(np.abs(stress).mean()),
                mae_change=float((np.abs(stress)-np.abs(cv)).mean()))


def summarize(rows, group_keys):
    report = []
    for keys, values in rows.groupby(group_keys, sort=True):
        groups = {name: metrics(group) for name, group in values.groupby('stratum')}
        wide = values.pivot(index=['block', 'match_id'], columns='stratum', values=['cv', 'anchor_speed', 'gap'])
        cv_difference = wide.cv.braking - wide.cv.stable
        report.append({**dict(zip(group_keys, keys)), **groups,
                       'cv_mean_braking_minus_stable': float(cv_difference.mean()),
                       'cv_median_paired_difference': float(cv_difference.median()),
                       'anchor_speed_mean_absolute_match_difference': float(
                           (wide.anchor_speed.braking-wide.anchor_speed.stable).abs().mean()),
                       'gap_mean_absolute_match_difference': float((wide.gap.braking-wide.gap.stable).abs().mean())})
    return report


