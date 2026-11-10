"""Frozen camera/LiDAR observations; no annotations or future frames are inputs.

Candidate footprints describe observed returns, not inferred full vehicle boxes.
The policy/tracker owns retention thresholds, velocity inference and ego offset.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

CAMERA_CACHE_BUDGET = 256 * 1024**2
CAMERA_SCORE_FLOOR = .05
VEHICLE_LABELS = {3: "car", 6: "bus", 8: "truck"}
MODEL_NAME = "torchvision.ssdlite320_mobilenet_v3_large.COCO_V1"


@dataclass(frozen=True)
class GeometryConfig:
    x_min: float = 2.
    x_max: float = 65.
    lateral_limit: float = 12.
    ground_cell_m: float = 2.
    ground_inlier_m: float = .18
    ground_max_slope: float = .20
    ransac_trials: int = 64
    height_min: float = .30
    height_max: float = 3.5
    bev_cell_m: float = .50
    min_points: int = 8
    max_longitudinal_extent: float = 16.
    max_lateral_extent: float = 4.5
    association_min_iou: float = .02
    association_min_coverage: float = .30
    max_camera_lag_s: float = .20


GEOMETRY = GeometryConfig()


def get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, allow_nan=False)+"\n").encode()
    temporary = path.with_suffix(path.suffix+".pending")
    temporary.write_bytes(encoded)
    temporary.replace(path)


class CameraDetector:
    """One cached float32 inference per image with official frozen COCO weights."""

    def __init__(self, cache_root, device="auto"):
        import torch
        import torchvision
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
        self.torch = torch
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._budget(15*1024**2)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        weight_dir = self.root / "weights"
        state = torch.hub.load_state_dict_from_url(weights.url, model_dir=str(weight_dir),
                                                 map_location="cpu", check_hash=True, progress=False)
        self.weight_path = weight_dir / Path(weights.url).name
        self.weight_hash = sha256(self.weight_path)
        self.model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        self.model.load_state_dict(state, strict=True)
        del state
        self.transform = weights.transforms()
        self.device = "mps" if device == "auto" and torch.backends.mps.is_available() else ("cpu" if device == "auto" else device)
        if self.device not in ("cpu", "mps"):
            raise ValueError("Only CPU or MPS inference is allowed")
        self.model.eval().to(self.device)
        self.signature = dict(model=MODEL_NAME, weights_sha256=self.weight_hash, torch=torch.__version__,
                              torchvision=torchvision.__version__, score_floor=CAMERA_SCORE_FLOOR,
                              vehicle_labels=VEHICLE_LABELS, dtype="float32")
        self.signature_hash = hashlib.sha256(json.dumps(self.signature, sort_keys=True).encode()).hexdigest()
        self.manifest = dict(**self.signature, source_url=weights.url, cache_budget_bytes=CAMERA_CACHE_BUDGET,
                             source_sha256=sha256(__file__), requested_device=device, actual_device=self.device,
                             inference_signature_sha256=self.signature_hash, training="none")
        write_json(self.root / "model_manifest.json", self.manifest)

    def _budget(self, additional=0):
        size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if size + additional > CAMERA_CACHE_BUDGET:
            raise RuntimeError("Camera cache total-size budget exceeded; preserve existing evidence and stop")

    def infer(self, image_path, source_id=None):
        from PIL import Image
        started = time.perf_counter()
        image_path = Path(image_path)
        image_hash = sha256(image_path)
        cache_key = hashlib.sha256((image_hash+self.signature_hash).encode()).hexdigest()
        target = self.root / "predictions" / f"{cache_key}.json"
        if target.exists():
            result = json.loads(target.read_text())
            if result["image_sha256"] != image_hash or result["inference_signature_sha256"] != self.signature_hash:
                raise ValueError("Camera cache provenance mismatch")
            return dict(result, cache_hit=True, source_id=source_id or str(image_path))
        self._budget(64*1024)
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            tensor = self.transform(image).to(self.device)
        failure = None
        forward_started = time.perf_counter()
        try:
            with self.torch.inference_mode():
                prediction = self.model([tensor])[0]
            if self.device == "mps":
                self.torch.mps.synchronize()
        except (RuntimeError, NotImplementedError) as error:
            if self.device != "mps":
                raise
            failure = str(error)[:500]
            self.device = "cpu"
            self.model.to("cpu")
            with self.torch.inference_mode():
                prediction = self.model([tensor.to("cpu")])[0]
            self.manifest.update(actual_device="cpu", mps_fallback_reason=failure)
            write_json(self.root / "model_manifest.json", self.manifest)
        forward_ms = 1000*(time.perf_counter()-forward_started)
        boxes, labels, scores = (prediction[k].detach().cpu().numpy() for k in ("boxes", "labels", "scores"))
        detections = [dict(box_xyxy=b.tolist(), score=float(s), class_id=int(c), class_name=VEHICLE_LABELS[int(c)])
                      for b, c, s in zip(boxes, labels, scores) if int(c) in VEHICLE_LABELS and s >= CAMERA_SCORE_FLOOR]
        result = dict(source_id=source_id or str(image_path), image_path=str(image_path), image_sha256=image_hash,
                      inference_signature_sha256=self.signature_hash, weights_sha256=self.weight_hash,
                      width=width, height=height, detections=detections, device=self.device,
                      forward_ms=forward_ms, total_ms=1000*(time.perf_counter()-started),
                      mps_fallback_reason=failure, cache_hit=False)
        write_json(target, result)
        return result

    def smoke(self, image_paths):
        paths = list(dict.fromkeys(map(str, image_paths)))[:10]
        if len(paths) != 10:
            raise ValueError("Smoke requires ten distinct real image paths")
        results = [self.infer(p) for p in paths]
        timings = np.array([r["forward_ms"] for r in results])
        report = dict(n_images=10, image_paths=paths, device=self.device, weights_sha256=self.weight_hash,
                      source_sha256=sha256(__file__), inference_signature_sha256=self.signature_hash,
                      first_forward_ms=float(timings[0]), mean_forward_ms=float(timings.mean()),
                      median_forward_ms=float(np.median(timings)), p90_forward_ms=float(np.quantile(timings, .9)),
                      n_cache_hits=sum(r["cache_hit"] for r in results),
                      n_vehicle_detections=[len(r["detections"]) for r in results],
                      semantics="Real-image runtime smoke; not detector or risk accuracy evidence")
        key = hashlib.sha256("\n".join(paths).encode()).hexdigest()[:12]
        write_json(self.root / f"smoke_{key}.json", report)
        return report


def transform_points(points, matrix):
    points, matrix = np.asarray(points, dtype=float), np.asarray(matrix, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def estimate_ground(points, config=GEOMETRY):
    """Deterministic lower-envelope RANSAC; only the current LiDAR sweep."""
    cells = np.floor(points[:, :2]/config.ground_cell_m).astype(int)
    minima = {}
    for index, cell in enumerate(map(tuple, cells)):
        if cell not in minima or points[index, 2] < points[minima[cell], 2]:
            minima[cell] = index
    lows = points[list(minima.values())]
    fallback = np.array([0., 0., float(np.quantile(points[:, 2], .10))])
    if len(lows) < 6:
        return fallback, 0.
    design = np.column_stack([lows[:, :2], np.ones(len(lows))])
    rng, best, best_mask = np.random.default_rng(0), fallback, np.zeros(len(lows), dtype=bool)
    for _ in range(config.ransac_trials):
        chosen = rng.choice(len(lows), 3, replace=False)
        try:
            plane = np.linalg.solve(design[chosen], lows[chosen, 2])
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(plane[:2]) > config.ground_max_slope:
            continue
        mask = np.abs(design @ plane-lows[:, 2]) < config.ground_inlier_m
        if mask.sum() > best_mask.sum():
            best, best_mask = plane, mask
    if best_mask.sum() >= 6:
        refined = np.linalg.lstsq(design[best_mask], lows[best_mask, 2], rcond=None)[0]
        if np.linalg.norm(refined[:2]) <= config.ground_max_slope:
            best = refined
    return best, float(best_mask.mean())


def lidar_clusters(xyz_ego, config=GEOMETRY):
    points = np.asarray(xyz_ego, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz_ego must be N×3")
    keep = np.isfinite(points).all(axis=1) & (points[:, 0] >= config.x_min) & (points[:, 0] <= config.x_max)
    keep &= (np.abs(points[:, 1]) <= config.lateral_limit) & (points[:, 2] >= -5.) & (points[:, 2] <= 5.)
    points = points[keep]
    if len(points) < config.min_points:
        return [], dict(ground_plane=None, ground_quality=0., n_roi_points=len(points))
    plane, ground_quality = estimate_ground(points, config)
    height = points[:, 2] - (points[:, :2] @ plane[:2]+plane[2])
    points = points[(height >= config.height_min) & (height <= config.height_max)]
    cells = {}
    for index, cell in enumerate(map(tuple, np.floor(points[:, :2]/config.bev_cell_m).astype(int))):
        cells.setdefault(cell, []).append(index)
    unseen, groups = set(cells), []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        pending, indices = [seed], []
        while pending:
            cell = pending.pop()
            indices.extend(cells[cell])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (cell[0]+dx, cell[1]+dy)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        pending.append(neighbor)
        if len(indices) >= config.min_points:
            groups.append(points[indices])
    candidates = []
    for cluster in groups:
        lower, upper = np.quantile(cluster, [.02, .98], axis=0)
        extent = upper-lower
        if not (.3 <= extent[0] <= config.max_longitudinal_extent and .2 <= extent[1] <= config.max_lateral_extent and .2 <= extent[2] <= 3.5):
            continue
        center = .5*(lower+upper)
        quality = min(1., len(cluster)/30.) * min(1., extent[0]/1.5) * min(1., extent[1]/.7) * min(1., extent[2]/.5)
        candidates.append(dict(center_ego=center.tolist(), footprint_ego_xyxy=[float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])],
                               observed_rear_x_ego_m=float(lower[0]), observed_extent_xyz=extent.tolist(),
                               npoints=len(cluster), geometry_quality=float(quality), points_ego=cluster))
    return candidates, dict(ground_plane=plane.tolist(), ground_quality=ground_quality, n_roi_points=int(keep.sum()), n_nonground_points=len(points))


def project_to_camera(points, frame, config=GEOMETRY):
    camera = get(frame, "camera")
    lag = (get(frame, "lidar_timestamp_ns")-get(camera, "timestamp_ns"))*1e-9
    if lag < -1e-9:
        raise ValueError("Future camera frame is forbidden")
    if lag > config.max_camera_lag_s:
        return np.empty((0, 2)), np.zeros(len(points), dtype=bool)
    matrix = np.linalg.inv(get(camera, "ego_T_camera")) @ np.linalg.inv(get(camera, "world_T_ego_camera")) @ get(frame, "world_T_ego_lidar")
    camera_points = transform_points(points, matrix)
    homogeneous = camera_points @ np.asarray(get(camera, "K")).T
    uv = homogeneous[:, :2] / np.where(np.abs(homogeneous[:, 2:3]) > 1e-9, homogeneous[:, 2:3], np.nan)
    valid = np.isfinite(uv).all(axis=1) & (camera_points[:, 2] > 0)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < get(camera, "width")) & (uv[:, 1] >= 0) & (uv[:, 1] < get(camera, "height"))
    return uv, valid


def associate_camera(candidates, detections, frame, config=GEOMETRY):
    """Greedy one-to-one geometric association; scores remain uncalibrated."""
    pairs = []
    for index, candidate in enumerate(candidates):
        candidate.update(camera_score=0., camera_class=None, camera_box_xyxy=None, camera_iou=0., camera_coverage=0.)
        if get(frame, "camera") is None:
            continue
        uv, valid = project_to_camera(candidate["points_ego"], frame, config)
        if valid.sum() < 3:
            continue
        pixels = uv[valid]
        lo, hi = np.quantile(pixels, [.02, .98], axis=0)
        area = np.prod(np.maximum(hi-lo, 0.))
        for det_index, det in enumerate(detections):
            box = np.asarray(det["box_xyxy"])
            intersection = np.prod(np.maximum(0., np.minimum(hi, box[2:])-np.maximum(lo, box[:2])))
            iou = intersection/max(1e-9, area+np.prod(box[2:]-box[:2])-intersection)
            coverage = np.mean(((pixels >= box[:2]) & (pixels <= box[2:])).all(axis=1))
            if iou >= config.association_min_iou and coverage >= config.association_min_coverage:
                pairs.append((float(det["score"])*float(coverage), index, det_index, float(iou), float(coverage)))
    used_candidates, used_detections = set(), set()
    for _, index, det_index, iou, coverage in sorted(pairs, reverse=True):
        if index in used_candidates or det_index in used_detections:
            continue
        used_candidates.add(index)
        used_detections.add(det_index)
        det = detections[det_index]
        candidates[index].update(camera_score=float(det["score"]), camera_class=det["class_name"],
                                  camera_box_xyxy=det["box_xyxy"], camera_iou=iou, camera_coverage=coverage)


def perceive_frame(frame, detector: CameraDetector | None, config=GEOMETRY):
    """Return candidates and diagnostics. The frame contract has no GT member."""
    camera = get(frame, "camera")
    if camera is not None and get(camera, "timestamp_ns") > get(frame, "lidar_timestamp_ns"):
        raise ValueError("Future camera input is forbidden")
    candidates, diagnostics = lidar_clusters(get(frame, "xyz_ego"), config)
    prediction = detector.infer(get(camera, "path")) if camera is not None and detector is not None else None
    if prediction is not None and (prediction["width"] != get(camera, "width") or prediction["height"] != get(camera, "height")):
        raise ValueError("Image dimensions disagree with calibration")
    associate_camera(candidates, prediction["detections"] if prediction else [], frame, config)
    for index, candidate in enumerate(candidates):
        candidate.pop("points_ego")
        xmin, ymin, xmax, ymax = candidate["footprint_ego_xyxy"]
        z = candidate["center_ego"][2]
        corners = np.array([[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]])
        candidate.update(candidate_id=f"{get(frame, 'log_id')}:{get(frame, 'lidar_timestamp_ns')}:{index}",
                         center_world=transform_points(np.asarray(candidate["center_ego"])[None], get(frame, "world_T_ego_lidar"))[0].tolist(),
                         footprint_world_xyz=transform_points(corners, get(frame, "world_T_ego_lidar")).tolist(),
                         lidar_timestamp_ns=get(frame, "lidar_timestamp_ns"), camera_timestamp_ns=get(camera, "timestamp_ns") if camera else None,
                         lidar_source_id=str(get(frame, "lidar_path")), camera_source_id=str(get(camera, "path")) if camera else None)
    diagnostics.update(geometry_config=asdict(config), camera_model=MODEL_NAME if prediction else None,
                       camera_weights_sha256=prediction["weights_sha256"] if prediction else None,
                       camera_lag_s=(get(frame, "lidar_timestamp_ns")-get(camera, "timestamp_ns"))*1e-9 if camera else None,
                       n_camera_vehicles=len(prediction["detections"]) if prediction else 0,
                       camera_cache_hit=prediction["cache_hit"] if prediction else None)
    return candidates, diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ten-real-image camera runtime smoke only")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("images", nargs=10, type=Path)
    args = parser.parse_args()
    print(json.dumps(CameraDetector(args.cache_root, args.device).smoke(args.images), indent=2))
