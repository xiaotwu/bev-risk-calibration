"""Analyze measured JSONL gate logs; test fixtures must be labeled explicitly."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def correlation(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze(rows, expected_tokens, baseline):
    if not rows:
        raise ValueError('No gate records')
    tokens = [r['sample_token'] for r in rows]
    if len(set(tokens)) != len(tokens) or set(tokens) != set(expected_tokens):
        raise ValueError('Duplicate, missing, or unexpected sample tokens')
    if len(set(expected_tokens)) != len(expected_tokens):
        raise ValueError('Expected token list contains duplicates')
    baseline = np.asarray(baseline, dtype=float)
    if baseline.ndim != 1 or not len(baseline) or not np.isfinite(baseline).all() or ((baseline < 0) | (baseline > 1)).any():
        raise ValueError('Invalid baseline coefficients')
    keys = [f'gate_layer_{i+1}' for i in range(len(baseline))]
    if any(set(k for k in r if k.startswith('gate_layer_')) != set(keys) for r in rows):
        raise ValueError('Inconsistent layer count')
    if any(not isinstance(r.get('scene_token'), str) or not r['scene_token'] for r in rows):
        raise ValueError('Missing scene token')
    w = np.asarray([[r[k] for k in keys] for r in rows], dtype=float)
    density = np.asarray([r['lidar_in_range_point_count'] for r in rows], dtype=float)
    if not np.isfinite(w).all() or ((w < 0) | (w > 1)).any() or not np.isfinite(density).all() or (density < 0).any():
        raise ValueError('Nonfinite/out-of-range measurements')
    definitions = {json.dumps([r.get('point_count_definition'), r.get('in_range_definition'), r.get('point_cloud_range')]) for r in rows}
    if len(definitions) != 1 or any(r.get('point_cloud_range') is None or not r.get('in_range_definition') for r in rows):
        raise ValueError('Missing or mixed density definitions')
    scenes = sorted(set(r['scene_token'] for r in rows))
    scene_means = np.asarray([w[[r['scene_token'] == s for r in rows]].mean(axis=0) for s in scenes])
    summary = []
    for i in range(w.shape[1]):
        x = w[:, i]
        summary.append(dict(layer=i+1, baseline=float(baseline[i]), mean=float(x.mean()),
            variance=float(x.var()), std=float(x.std()), median=float(np.median(x)),
            q25=float(np.quantile(x,.25)), q75=float(np.quantile(x,.75)),
            minimum=float(x.min()), maximum=float(x.max()),
            mean_abs_baseline_deviation=float(np.abs(x-baseline[i]).mean()),
            saturation_fraction=float(((x < .05) | (x > .95)).mean()),
            scene_weighted_mean=float(scene_means[:,i].mean()),
            density_pearson=correlation(density,x),
            density_spearman=correlation(rankdata(density),rankdata(x))))
    corr = [[correlation(w[:,i],w[:,j]) for j in range(w.shape[1])] for i in range(w.shape[1])]
    return dict(sample_count=len(rows),scene_count=len(scenes),layers=summary,
                pearson_matrix=corr,uncertainty='No confidence intervals; frames are temporally dependent',
                interpretation='Associations do not establish reliability or detection benefit')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',required=True)
    p.add_argument('--expected-tokens',required=True,help='JSON list from evaluated dataset, not from the gate log')
    p.add_argument('--baseline',required=True,help='JSON list of camera coefficients in layer order')
    p.add_argument('--output',required=True)
    p.add_argument('--fixture',action='store_true',help='Label every artifact as a software fixture')
    a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.records).read_text().splitlines() if x.strip()]
    expected=json.loads(Path(a.expected_tokens).read_text()); baseline=json.loads(Path(a.baseline).read_text())
    result=analyze(rows,expected,baseline)
    result['evidence_type']='software_fixture_not_model_results' if a.fixture else 'provided_gate_log_analysis_not_independently_reproduced'
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (out/'gate_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['layers'][0]));writer.writeheader();writer.writerows(result['layers'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    w=np.asarray([[r[f'gate_layer_{i+1}'] for i in range(len(baseline))] for r in rows])
    label='SOFTWARE FIXTURE - NOT MODEL RESULTS' if a.fixture else 'Recorded gate outputs'
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for i in range(w.shape[1]):
        axes[0,0].hist(w[:,i],bins=np.linspace(0,1,21),histtype='step',label=f'Layer {i+1}')
        axes[1,1].scatter([r['lidar_in_range_point_count'] for r in rows],w[:,i],s=8,alpha=.5,label=f'Layer {i+1}')
    axes[0,0].set(xlabel='Camera coefficient',ylabel='Frame count');axes[0,0].legend()
    axes[0,1].plot(range(1,len(baseline)+1),w.mean(0),'o-',label='Frame-weighted mean')
    axes[0,1].plot(range(1,len(baseline)+1),baseline,'x--',label='Baseline')
    axes[0,1].set(xlabel='Encoder layer',ylabel='Camera coefficient',ylim=(0,1));axes[0,1].legend()
    arr=np.asarray([[np.nan if v is None else v for v in row] for row in result['pearson_matrix']])
    cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('lightgray')
    im=axes[1,0].imshow(arr,vmin=-1,vmax=1,cmap=cmap)
    axes[1,0].set(xticks=range(len(baseline)),yticks=range(len(baseline)),xticklabels=range(1,len(baseline)+1),yticklabels=range(1,len(baseline)+1),xlabel='Layer',ylabel='Layer',title='Pearson (gray = undefined)');fig.colorbar(im,ax=axes[1,0])
    axes[1,1].set(xlabel='In-range points before voxelization',ylabel='Camera coefficient',ylim=(0,1))
    fig.suptitle(f'{label}; N={len(rows)}, scenes={result["scene_count"]}')
    fig.tight_layout();fig.savefig(out/'gate_analysis.png',dpi=160);fig.savefig(out/'gate_analysis.pdf');plt.close(fig)
    print(json.dumps({'output':str(out),'sample_count':len(rows),'evidence_type':result['evidence_type']}))


if __name__=='__main__':main()
