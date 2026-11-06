# Preserve the detector; replace only the original machine-specific data paths.
_base_ = ['./bevformer_project.py']
data_root = 'data/nuscenes/'
data = dict(
    train=dict(data_root=data_root, ann_file=data_root+'nuscenes_infos_temporal_train.pkl'),
    val=dict(data_root=data_root, ann_file=data_root+'nuscenes_infos_temporal_val.pkl'),
    test=dict(data_root=data_root, ann_file=data_root+'nuscenes_infos_temporal_val.pkl'))
