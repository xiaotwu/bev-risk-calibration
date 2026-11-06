# Opt-in research config; full detector validation pending.
_base_ = ['./bevformer_reproduction.py']
model = dict(pts_bbox_head=dict(transformer=dict(encoder_gate_mode='fixed_refactored')))
