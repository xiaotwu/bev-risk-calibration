"""Scene-conditioned encoder fusion. Full detector validation remains pending."""
import torch
from torch import nn


class LayerGate(nn.Module):
    """One camera-preference weight per sample/layer.

    Explicit contract: camera levels [B,V,C,H,W] with common C;
    LiDAR [B,D,H,W]. Equal weight per view and FPN level, no padding mask.
    Detach applies BEFORE trainable projections so they can still learn.
    """
    def __init__(self, camera_channels, lidar_channels, baseline_logits,
                 summary_dim=64, hidden_dim=64, detach_features=False):
        super().__init__()
        alpha = torch.as_tensor(baseline_logits, dtype=torch.float32).detach().clone()
        if alpha.ndim != 1 or alpha.numel() == 0 or not torch.isfinite(alpha).all():
            raise ValueError('baseline_logits must be a finite nonempty vector')
        if min(camera_channels, lidar_channels, summary_dim, hidden_dim) <= 0:
            raise ValueError('channel dimensions must be positive')
        self.detach_features = detach_features
        self.camera_channels, self.lidar_channels = camera_channels, lidar_channels
        self.camera_projection = nn.Linear(camera_channels, summary_dim)
        self.lidar_projection = nn.Linear(lidar_channels, summary_dim)
        self.norm = nn.LayerNorm(2 * summary_dim)
        self.hidden = nn.Linear(2 * summary_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, alpha.numel())
        self.register_buffer('baseline_logits', alpha)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(alpha)

    def forward(self, camera, lidar):
        levels = [camera] if isinstance(camera, torch.Tensor) else list(camera)
        if not levels or lidar.ndim != 4 or lidar.shape[1] != self.lidar_channels:
            raise ValueError('expected nonempty FPN and LiDAR [B,D,H,W]')
        b = lidar.shape[0]
        if min(lidar.shape) == 0:
            raise ValueError('empty LiDAR feature axis')
        pooled = []
        for x in levels:
            if x.ndim != 5 or x.shape[0] != b or x.shape[2] != self.camera_channels or min(x.shape) == 0:
                raise ValueError('camera level must be [B,V,C,H,W] with configured C')
            if x.device != lidar.device:
                raise ValueError('sensor features must share a device')
            if self.detach_features:
                x = x.detach()
            # Float32 accumulation avoids half precision pooling overflow.
            pooled.append(x.float().mean(dim=(1, 3, 4)))
        y = lidar.detach() if self.detach_features else lidar
        cam = torch.stack(pooled).mean(dim=0).to(self.camera_projection.weight.dtype)
        lid = y.float().mean(dim=(2, 3)).to(self.lidar_projection.weight.dtype)
        s = torch.cat((self.camera_projection(cam), self.lidar_projection(lid)), dim=-1)
        return torch.sigmoid(self.output(self.activation(self.hidden(self.norm(s)))))


def fuse(camera, lidar, weights, layer, batch_dim=0):
    """Explicit batch axis, supports [B,Q,C] and [Q,B,C] without inference."""
    if camera.shape != lidar.shape or camera.ndim < 2:
        raise ValueError('modality outputs must have identical shapes')
    if camera.device != lidar.device or camera.dtype != lidar.dtype:
        raise ValueError('modality outputs must share device and dtype')
    if batch_dim not in range(camera.ndim) or weights.ndim != 2:
        raise ValueError('invalid batch axis or weights shape')
    if weights.shape[0] != camera.shape[batch_dim] or not 0 <= layer < weights.shape[1]:
        raise ValueError('invalid batch size or layer index')
    shape = [1] * camera.ndim
    shape[batch_dim] = weights.shape[0]
    w = weights[:, layer].to(device=camera.device, dtype=camera.dtype).reshape(shape)
    return w * camera + (1 - w) * lidar


def load_baseline_strict(model, state_dict, gate_prefix='layer_gate.'):
    """For additive integration retaining legacy scalars as frozen parameters.

    Caller must extract actual logits and construct the gate first. No heuristic
    key renaming or silent dropping. Validate all keys/shapes before mutation.
    """
    expected = model.state_dict()
    missing = set(expected) - set(state_dict)
    extra = set(state_dict) - set(expected)
    new = {k for k in expected if k.startswith(gate_prefix)}
    bad_shapes = [k for k in set(expected) & set(state_dict)
                  if expected[k].shape != state_dict[k].shape]
    if missing != new or extra or bad_shapes:
        raise ValueError(f'checkpoint mismatch: missing={sorted(missing)}, extra={sorted(extra)}, shapes={bad_shapes}')
    return model.load_state_dict(state_dict, strict=False)


def parameter_count(camera_channels, lidar_channels, layers, summary_dim=64, hidden_dim=64):
    d, h = summary_dim, hidden_dim
    return d * (camera_channels + lidar_channels + 2) + 4*d + (2*d+1)*h + (h+1)*layers
