"""Prepare a legacy Enc_Dec checkpoint for the audited LayerGate config.

Does not prove full-model compatibility; perform the documented parity gate next.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('layer_gate',ROOT/'projects/mmdet3d_plugin/bevformer/modules/layer_gate.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
PREFIX='pts_bbox_head.transformer.'

def convert(checkpoint):
    if not isinstance(checkpoint,dict) or 'state_dict' not in checkpoint:
        raise ValueError('Expected an MMCV checkpoint with state_dict; no heuristic key remapping')
    state=dict(checkpoint['state_dict'])
    if any(k.startswith(PREFIX+'layer_gate.') for k in state):
        raise ValueError('Already contains a gate; refusing to reset learned dynamic weights')
    keys=[PREFIX+f'encoder.layers.{i}.cross_model_weights_logit' for i in range(4)]
    found=sorted(k for k in state if k.endswith('cross_model_weights_logit'))
    if found != keys or any(state[k].numel()!=1 for k in keys):
        raise ValueError('Expected exactly four scalar logits with audited unwrapped keys')
    alpha=torch.stack([state[k].float().reshape(()) for k in keys])
    # The source config specifies camera=256, raw pillar BEV=64.
    gate=mod.LayerGate(256,64,alpha)
    state.update({PREFIX+'layer_gate.'+k:v for k,v in gate.state_dict().items()})
    result={k:v for k,v in checkpoint.items() if k not in ('optimizer','state_dict')}
    result['state_dict']=state
    result['layergate_conversion']={'source_logits':alpha.tolist(),'source_camera_weights':alpha.sigmoid().tolist(),
        'added_parameters':sum(p.numel() for p in gate.parameters()),
        'instruction':'Use load_from for new fine-tuning; never resume_from the legacy optimizer/iteration.'}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('output');a=p.parse_args()
    target=Path(a.output)
    if target.exists(): raise FileExistsError(target)
    torch.manual_seed(0)
    result=convert(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    result['layergate_conversion']['source_sha256']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,target)
    print(result['layergate_conversion'])
