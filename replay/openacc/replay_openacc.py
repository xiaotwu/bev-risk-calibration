"""Predeclared same-run OpenACC AstaZero observation-age trajectory replay."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN_IDS = {f'ASta_040719_platoon{i}' for i in range(3, 11)} | {
    'ASta_050719_platoon1', 'ASta_050719_platoon2'}
TRACKS = ['Time', 'Speed1', 'Speed2', 'IVS1']
AGES = (1, 2, 3)
HISTORY = 6
FUTURE = 10 * max(AGES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for data in iter(lambda: stream.read(1 << 20), b''):
            digest.update(data)
        return digest.hexdigest()


def write_json(path, data):
    with path.open('x', encoding='utf8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def read_run(item, data_root):
    path = (data_root / item['path']).resolve(strict=True)
    require(path.is_relative_to(data_root / 'openacc' / 'raw'), 'Run file is outside the input directory')
    require(sha(path) == item['sha256'], f"Run hash differs for {item['id']}")
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        metadata = {row[0]: row[1:] for row in [next(reader) for _ in range(5)]}
    require(set(metadata) == {'Date', 'Vehicle_order', 'Number_of_vehicles', 'ACC',
                             'Distance_setting'}, f"Unexpected metadata {item['id']}")
    frame = pd.read_csv(path, skiprows=5, usecols=TRACKS, compression='gzip',
                        dtype={key: 'float64' for key in TRACKS})
    require(len(frame) == item['rows'], f"Run row count differs for {item['id']}")
    time = frame.Time.to_numpy()
    require(np.isfinite(time).all(), f"Nonfinite time in {item['id']}")
    ticks = np.rint(time * 10).astype(np.int64)
    require(np.max(np.abs(ticks - 10 * time)) < 1e-5 and np.all(np.diff(ticks) > 0),
            f"Off-grid, duplicated or reversed time in {item['id']}")
    return frame, ticks, metadata


def eligible(frame, ticks):
    lead = frame.Speed1.to_numpy()
    follower = frame.Speed2.to_numpy()
    gap = frame.IVS1.to_numpy()
    valid = (np.isfinite(lead) & np.isfinite(follower) & np.isfinite(gap) &
             (lead >= 0.) & (lead <= 55.) & (follower >= 0.) &
             (follower <= 55.) & (gap > 0.))
    edges = np.r_[0, np.cumsum(~valid)]
    base = np.arange(HISTORY, len(frame) - FUTURE)
    base = base[(ticks[base] % 5 == 0) & (gap[base] >= 2.) & (gap[base] <= 100.)]
    continuous = ticks[base+FUTURE] - ticks[base-HISTORY] == HISTORY + FUTURE
    complete = edges[base+FUTURE+1] == edges[base-HISTORY]
    anchors = base[continuous & complete]
    past = lead[anchors[:, None] + np.arange(-HISTORY, 1)]
    early = lead[anchors[:, None] + np.arange(HISTORY + 1)]
    delta = np.trapz(early - past, dx=1/HISTORY, axis=1)
    changes = dict(zip(anchors.tolist(), delta.tolist()))
    groups = (anchors[delta <= -.75], anchors[np.abs(delta) <= .25])
    audit = dict(raw_rows=len(frame), invalid_rows=int((~valid).sum()),
                 missing_100ms_steps=int((np.diff(ticks)-1).sum()), grid_with_gap=len(base),
                 valid_windows=len(anchors), raw_early_braking=len(groups[0]),
                 raw_early_stable=len(groups[1]))
    return groups, changes, audit


def select(frame, ticks, groups):
    braking, stable = groups
    speed = frame.Speed1.to_numpy()
    gap = frame.IVS1.to_numpy()
    free = np.ones(len(frame), dtype=bool)
    match_id = 0
    picks = []

    def occupy(row):
        lo = np.searchsorted(ticks, ticks[row] - HISTORY - FUTURE, side='left')
        hi = np.searchsorted(ticks, ticks[row] + HISTORY + FUTURE, side='right')
        free[lo:hi] = False

    for row in braking:
        if not free[row]:
            continue
        ds = abs(speed[stable] - speed[row])
        dg = abs(gap[stable] - gap[row])
        options = ((ds <= 3.) & (dg <= 10.) & free[stable] &
                   (abs(ticks[stable] - ticks[row]) > HISTORY + FUTURE))
        choice = np.flatnonzero(options)
        if len(choice):
            best = choice[np.argmin(ds[choice]/3. + dg[choice]/10.)]
            peer = stable[best]
            picks.extend([(row, 'braking', match_id), (peer, 'stable', match_id)])
            occupy(row)
            occupy(peer)
            match_id += 1

    for label, group in zip(('braking', 'stable'), groups):
        for row in group:
            if free[row]:
                picks.append((row, label, None))
                occupy(row)
    picks.sort()
    pick_ticks = ticks[[row for row, _, _ in picks]]
    require(np.all(np.diff(pick_ticks) > HISTORY+FUTURE), 'Selected windows overlap')
    return picks, match_id


def stress_shift(speed, age):
    horizon = max(0., age - 1.)
    active = np.minimum(horizon, speed/3.)
    return speed*(horizon-active) + 1.5*active**2, 3.*active


def score(frame, ticks, run_id, metadata, picks, changes):
    if not picks:
        return []
    lead = frame.Speed1.to_numpy()
    follower = frame.Speed2.to_numpy()
    gap = frame.IVS1.to_numpy()
    rows = np.asarray([row for row, _, _ in picks], dtype=np.int64)
    out = []
    for age in AGES:
        window = rows[:, None] + np.arange(age * 10 + 1)
        future = window[:, -1]
        require(np.all(ticks[future] - ticks[rows] == age*10), 'Wrong scoring interval')
        lead_advance = np.trapz(lead[window], dx=.1, axis=1)
        ego_advance = np.trapz(follower[window], dx=.1, axis=1)
        actual_advance = gap[future] - gap[rows] + ego_advance
        speed = lead[rows]
        speed_future = lead[future]
        position = speed*age - actual_advance
        stopping = (speed**2 - speed_future**2) / 16.
        cv = position + stopping
        direct = gap[rows] + speed*age - ego_advance + speed**2/16. - (gap[future] + speed_future**2/16.)
        require(np.allclose(cv, direct, rtol=0., atol=1e-10), 'B decomposition differs')
        closure = actual_advance - lead_advance
        gap_deficit, speed_deficit = stress_shift(speed, age)
        shift = -gap_deficit + ((speed-speed_deficit)**2 - speed**2)/16.
        corrected = cv + shift
        require(np.all(shift <= 1e-10) and (age != 1 or np.array_equal(cv, corrected)),
                'Stress changed the one-second baseline or increased B')
        for i, (row, label, match_id) in enumerate(picks):
            out.append(dict(run_id=run_id, age_s=age, anchor_s=ticks[row]/10., cohort=label,
                            match_id=match_id, follower_acc=metadata['ACC'][0],
                            distance_setting=metadata['Distance_setting'][0],
                            gap0_m=gap[row], lead_speed0_mps=speed[i], early_change_mps=changes[row],
                            gap_truth_m=gap[future[i]], lead_speed_truth_mps=speed_future[i],
                            cv_gap_error_m=position[i], cv_stop_error_m=stopping[i],
                            cv_B_error_m=cv[i], stress_B_error_m=corrected[i],
                            cv_B_doppler_travel_m=cv[i]+closure[i],
                            stress_B_doppler_travel_m=corrected[i]+closure[i],
                            gap_closure_m=closure[i]))
    return out


def metrics(rows, field):
    if not len(rows):
        return None
    values = rows[field].to_numpy()
    return dict(n=len(values), signed_mean_m=float(values.mean()),
                signed_median_m=float(np.median(values)), optimistic_share=float((values > 0.).mean()),
                absolute_mean_m=float(abs(values).mean()), positive_mean_m=float(np.maximum(values, 0).mean()),
                negative_mean_m=float(np.maximum(-values, 0).mean()))


def summaries(frame):
    result = {}
    for age in AGES:
        at_age = frame[frame.age_s.eq(age)]
        result[str(age)] = {}
        for name, sample in [('matched', at_age[at_age.match_id.notna()]), ('disjoint', at_age)]:
            by_group = {}
            for label, group in sample.groupby('cohort'):
                by_group[label] = dict(cv=metrics(group, 'cv_B_error_m'),
                                       stress=metrics(group, 'stress_B_error_m'),
                                       cv_doppler_travel=metrics(group, 'cv_B_doppler_travel_m'),
                                       stress_doppler_travel=metrics(group, 'stress_B_doppler_travel_m'),
                                       anchor_speed_mean_mps=float(group.lead_speed0_mps.mean()),
                                       anchor_gap_mean_m=float(group.gap0_m.mean()),
                                       gap_error_mean_m=float(group.cv_gap_error_m.mean()),
                                       stop_error_mean_m=float(group.cv_stop_error_m.mean()),
                                       gap_closure_abs_p95_m=float(np.quantile(abs(group.gap_closure_m), .95)))
            result[str(age)][name] = by_group
    return result


def run_level(by_run):
    output = {}
    for age in AGES:
        for subset in ('matched', 'disjoint'):
            fields = {}
            for run_id, record in by_run.items():
                groups = record['outcomes'][str(age)][subset]
                for label, item in groups.items():
                    for key, value in [
                            (f'{label}_cv_signed_m', item['cv']['signed_mean_m']),
                            (f'{label}_absolute_change_m', item['stress']['absolute_mean_m']-item['cv']['absolute_mean_m']),
                            (f'{label}_negative_change_m', item['stress']['negative_mean_m']-item['cv']['negative_mean_m'])]:
                        fields.setdefault(key, {})[run_id] = value
                if 'braking' in groups and 'stable' in groups:
                    fields.setdefault('braking_minus_stable_cv_signed_m', {})[run_id] = (
                        groups['braking']['cv']['signed_mean_m'] - groups['stable']['cv']['signed_mean_m'])
            output[f'{age}_{subset}'] = {key: dict(n_runs=len(values),
                mean=float(np.mean(list(values.values()))), minimum=min(values.values()),
                maximum=max(values.values()), positive_runs=sum(value > 0. for value in values.values()),
                by_run=values) for key, values in fields.items()}
    return output


def run(data_root, output):
    data_root = data_root.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    outputs = [output / 'events.csv', output / 'results.json']
    receipt = data_root / 'openacc' / 'raw' / 'receipt.json'
    info = json.loads(receipt.read_text(encoding='utf8'))
    items = info['runs']
    require(len(items) == len(RUN_IDS) and {item['id'] for item in items} == RUN_IDS,
            'Incomplete or duplicate campaign run receipt')
    by_run = {}
    all_rows = []
    for item in items:
        name = item['id']
        frame, ticks, metadata = read_run(item, data_root)
        groups, changes, audit = eligible(frame, ticks)
        picks, matches = select(frame, ticks, groups)
        rows = score(frame, ticks, name, metadata, picks, changes)
        all_rows.extend(rows)
        audit.update(metadata=metadata, matched_pairs=matches, selected_windows=len(picks))
        outcomes = summaries(pd.DataFrame(rows)) if rows else {
            str(age): {'matched': {}, 'disjoint': {}} for age in AGES}
        by_run[name] = dict(audit=audit, outcomes=outcomes)
    events = pd.DataFrame(all_rows)
    require(len(events) > 0 and not events.duplicated(['run_id', 'anchor_s', 'age_s']).any(),
            'No supported events or duplicate scored events')
    events['match_id'] = events.match_id.astype('Int64')
    outcome = dict(status='complete', replay_source_sha256=sha(Path(__file__)),
                   raw_receipt_sha256=sha(receipt), rows=len(events), runs=by_run,
                   descriptive_window_pool=summaries(events), descriptive_run_pool=run_level(by_run),
                   primary_age_s=2, independent_road_sample_count=1,
                   external_validation='controlled-track recorded-trajectory open-loop replay')
    events.to_csv(outputs[0], index=False, mode='x')
    outcome['events_sha256'] = sha(outputs[0])
    write_json(outputs[1], outcome)
    print(dict(status='complete', runs=len(by_run), events=len(events)//len(AGES),
               matched_pairs=sum(row['audit']['matched_pairs'] for row in by_run.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent)
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    run(args.data_root, args.output)
