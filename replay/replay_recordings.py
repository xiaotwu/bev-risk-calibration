"""Run the paper's NGSIM replay on the six original recording-clock partitions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from replay import ROOT, eligibility, match, ngsim_joins, score, sha, summarize


COUNTS = {
    ('i-80', 1113433134900): 1262678,
    ('i-80', 1113436766900): 1549918,
    ('i-80', 1113437564900): 1753791,
    ('us-101', 1118846978900): 1180598,
    ('us-101', 1118847842000): 2107095,
    ('us-101', 1118848757000): 1515240,
}


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def run(data_root, output):
    data_root = data_root.resolve(strict=True)
    receipt = data_root/'raw_recordings/receipt.json'
    blocks = json.loads(receipt.read_text(encoding='utf-8'))['blocks']
    if len(blocks) != 6 or {(b['site'], b['origin_ms']): b['rows'] for b in blocks} != COUNTS:
        raise ValueError('Recording inventory differs from official counts')
    if len({b['id'] for b in blocks}) != 6:
        raise ValueError('Recording identifiers are not unique')
    output.mkdir(parents=True, exist_ok=False)
    all_rows = []
    audits = []
    for block in blocks:
        frame, audit = ngsim_joins(block, data_root)
        clocks = pd.read_csv(data_root/block['path'], usecols=['global_time', 'frame_id'])
        if not clocks.global_time.sub(clocks.frame_id.mul(100)).eq(block['origin_ms']).all():
            raise ValueError(f"Recording-clock contamination in {block['id']}")
        del clocks
        braking, stable, eligible = eligibility(frame)
        pairs = match(braking, stable)
        audit.update(id=block['id'], site=block['site'], origin_ms=block['origin_ms'],
                     eligible_anchors=eligible, braking_candidates=len(braking),
                     stable_candidates=len(stable), matched_pairs=len(pairs))
        audits.append(audit)
        if len(pairs):
            all_rows.append(score(frame, pairs, block['id'], block['site']))
        print(audit, flush=True)
    rows = pd.concat(all_rows, ignore_index=True)
    rows.to_csv(output/'matched_scores.csv', index=False)
    summary = dict(status='complete', source_receipt_sha256=sha(receipt),
                   replay_source_sha256=sha(Path(__file__)),
                   scoring_source_sha256=sha(ROOT/'replay.py'),
                   matched_scores_sha256=sha(output/'matched_scores.csv'), audits=audits,
                   by_block=summarize(rows, ['site', 'block', 'velocity_source', 'age_s']),
                   by_site=summarize(rows, ['site', 'velocity_source', 'age_s']))
    (output/'summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=json_scalar) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'results_recordings')
    args = parser.parse_args()
    run(args.data_root, args.output)
