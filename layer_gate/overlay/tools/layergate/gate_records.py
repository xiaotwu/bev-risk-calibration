"""Validated, append-only gate records. No torch/MMCV dependency."""
import json
import math
from pathlib import Path


class GateRecorder:
    def __init__(self, path, expected_tokens, layers, point_cloud_range, token_to_scene=None):
        self.path = Path(path)
        self.expected = set(expected_tokens)
        if len(self.expected) != len(expected_tokens) or not self.expected:
            raise ValueError('Expected tokens must be unique and nonempty')
        if layers <= 0 or len(point_cloud_range) != 6:
            raise ValueError('Invalid layer count or point cloud range')
        self.token_to_scene = token_to_scene
        if token_to_scene is not None and set(token_to_scene) != self.expected:
            raise ValueError("Scene map must cover expected tokens exactly")
        self.layers = layers
        self.point_cloud_range = list(point_cloud_range)
        self.seen = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('x', encoding='utf-8')
        with self.path.with_suffix(self.path.suffix + '.expected_tokens.json').open('x') as f:
            json.dump(list(expected_tokens), f)

    def write(self, token, scene, weights, point_count, in_range_count):
        if not isinstance(token, str) or token not in self.expected:
            raise ValueError('Missing or unexpected sample token')
        if not isinstance(scene, str) or not scene:
            raise ValueError('Missing scene token')
        if self.token_to_scene is not None and self.token_to_scene[token] != scene:
            raise ValueError('Scene token does not match dataset metadata')
        if token in self.seen:
            raise ValueError('Duplicate sample token: ' + token)
        if len(weights) != self.layers or any(not math.isfinite(float(w)) or not 0 <= w <= 1 for w in weights):
            raise ValueError('Invalid gate vector')
        if not 0 <= in_range_count <= point_count:
            raise ValueError('Invalid point counts')
        row = dict(sample_token=token, scene_token=scene,
                   lidar_point_count=int(point_count), lidar_in_range_point_count=int(in_range_count),
                   point_count_definition='model_input_before_voxelization',
                   in_range_definition='xyz_lower_inclusive_upper_exclusive',
                   point_cloud_range=self.point_cloud_range)
        row.update({f'gate_layer_{i+1}': float(w) for i, w in enumerate(weights)})
        self.file.write(json.dumps(row, allow_nan=False) + '\n')
        self.file.flush()
        self.seen.add(token)

    def capture(self, detector, metas, points):
        """Called after successful simple_test; detached values only."""
        transformer = detector.pts_bbox_head.transformer
        mode = transformer.encoder_gate_mode
        if mode == 'fixed_legacy':
            weights = transformer._fixed_encoder_logits().sigmoid().detach().cpu().tolist()
            rows = [weights] * len(metas)
        else:
            weights = transformer.last_encoder_gates
            if weights is None:
                raise ValueError('No encoder gates: LiDAR fusion may not have executed')
            rows = weights.detach().float().cpu().tolist()
        pts_list = list(points) if isinstance(points, (list, tuple)) else [points]
        if len(rows) != len(metas) or len(pts_list) != len(metas):
            raise ValueError('Gate/metadata/point batch mismatch')
        for meta, pts, weights in zip(metas, pts_list, rows):
            while isinstance(pts, (list, tuple)):
                if len(pts) != 1:
                    raise ValueError('Ambiguous point augmentation layout')
                pts = pts[0]
            if hasattr(pts, 'tensor'):
                pts = pts.tensor
            if pts is None or pts.ndim != 2 or pts.shape[1] < 3:
                raise ValueError('Expected point matrix')
            low = pts.new_tensor(self.point_cloud_range[:3])
            high = pts.new_tensor(self.point_cloud_range[3:])
            count = int(((pts[:, :3] >= low) & (pts[:, :3] < high)).all(dim=1).sum().item())
            self.write(meta.get('sample_idx'), meta.get('scene_token'), weights, len(pts), count)

    def finalize(self):
        self.file.close()
        missing = self.expected - self.seen
        if missing:
            raise ValueError(f'Incomplete gate coverage: {len(missing)} samples missing')
        summary = dict(status='complete_gate_coverage', sample_count=len(self.seen),
                       layer_count=self.layers, point_cloud_range=self.point_cloud_range,
                       detection_metrics_verified=False)
        with self.path.with_suffix(self.path.suffix + '.coverage.json').open('x') as f:
            json.dump(summary, f, indent=2)
        return summary
