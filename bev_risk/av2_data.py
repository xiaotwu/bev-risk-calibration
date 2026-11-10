"""Stream AV2 sensors without exposing labels; reference annotations are opt-in.

Official AV2 points are already motion-compensated into ego at LiDAR time.
World coordinates here are the dataset's city coordinates. No pose interpolation
or future-camera selection is performed: exact supplied sensor-time poses are
required, and only the most recent camera at or before the sweep is used.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def transform_from_pose(row) -> np.ndarray:
    """Return parent_T_child from AV2 scalar-first unit-quaternion pose fields."""
    q = np.asarray([row[k] for k in ("qw", "qx", "qy", "qz")], dtype=np.float64)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid AV2 quaternion")
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [row[k] for k in ("tx_m", "ty_m", "tz_m")]
    if not np.isfinite(transform).all():
        raise ValueError("Invalid AV2 translation")
    return transform


def transform_points(transform: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply parent_T_child to a finite [N,3] array of child-frame points."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if transform.shape != (4, 4) or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Expected a 4x4 transform and [N,3] points")
    return xyz @ transform[:3, :3].T + transform[:3, 3]


@dataclass(frozen=True)
class CameraFrame:
    path: Path
    timestamp_ns: int
    world_T_ego_camera: np.ndarray
    ego_T_camera: np.ndarray
    K: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class SensorFrame:
    log_id: str
    lidar_timestamp_ns: int
    lidar_path: Path
    xyz_ego: np.ndarray
    intensity: np.ndarray
    world_T_ego_lidar: np.ndarray
    camera: CameraFrame | None
    ego_velocity_ego_mps: np.ndarray | None

    @property
    def timestamp_ns(self) -> int:
        return self.lidar_timestamp_ns


def camera_T_ego_lidar(frame: SensorFrame) -> np.ndarray | None:
    """Ego-motion compensation only; moving-object lag remains unresolved."""
    if frame.camera is None:
        return None
    return (np.linalg.inv(frame.camera.ego_T_camera)
            @ np.linalg.inv(frame.camera.world_T_ego_camera)
            @ frame.world_T_ego_lidar)


def project_lidar_to_camera(frame: SensorFrame, minimum_depth_m: float = 0.1):
    """Return pixels, positive-depth/in-image mask and camera optical depth.

    Uses the official AV2 pinhole geometry convention. Calibration's legacy k1,
    k2 and k3 values are not reapplied to the released pinhole imagery.
    """
    if frame.camera is None:
        return np.full((len(frame.xyz_ego), 2), np.nan), np.zeros(len(frame.xyz_ego), dtype=bool), np.full(len(frame.xyz_ego), np.nan)
    xyz_camera = transform_points(camera_T_ego_lidar(frame), frame.xyz_ego)
    depth = xyz_camera[:, 2]
    positive = depth > minimum_depth_m
    uv = np.full((len(xyz_camera), 2), np.nan)
    uv[positive] = (xyz_camera[positive] @ frame.camera.K.T)[:, :2] / depth[positive, None]
    valid = (positive & (uv[:, 0] >= 0) & (uv[:, 0] < frame.camera.width)
             & (uv[:, 1] >= 0) & (uv[:, 1] < frame.camera.height))
    return uv, valid, depth


class AV2Log:
    """A single frozen log. Construction reads poses/calibration, never labels.

    ``project_root`` contains data/manifests and data/av2_sensor. With default
    ``require_complete=True``, every frozen selected source object for this log
    must exist at its expected byte size. SHA verification belongs to acquisition
    receipts. Explicit ``False`` is only for development smoke checks while the
    authorized acquisition is still running; it never changes the timestamp list.
    """
    def __init__(self, project_root: str | Path, log_id: str, *, require_complete: bool = True):
        self.project_root = Path(project_root)
        self.log_id = log_id
        self.root = self.project_root / "data/av2_sensor/val" / log_id
        manifest = json.loads((self.project_root / "data/manifests/av2_first6_objects.json").read_text())
        self.objects = [item for item in manifest["objects"] if item["log_id"] == log_id]
        if not self.objects:
            raise ValueError("Log is not in the frozen six-log manifest")
        if require_complete:
            missing = [item["local_path"] for item in self.objects
                       if not (self.project_root / item["local_path"]).is_file()
                       or (self.project_root / item["local_path"]).stat().st_size != item["bytes"]]
            if missing:
                raise FileNotFoundError(f"Frozen log acquisition incomplete ({len(missing)} objects): {missing[0]}")
        self.lidar_paths = {int(Path(item["key"]).stem): self.project_root / item["local_path"]
                            for item in self.objects if "/sensors/lidar/" in item["key"]}
        self.camera_paths = {int(Path(item["key"]).stem): self.project_root / item["local_path"]
                             for item in self.objects if "/sensors/cameras/ring_front_center/" in item["key"]}
        self.lidar_timestamps_ns = tuple(sorted(self.lidar_paths))
        self.camera_timestamps_ns = tuple(sorted(self.camera_paths))
        poses = pd.read_feather(self.root / "city_SE3_egovehicle.feather")
        if poses["timestamp_ns"].duplicated().any():
            raise ValueError("Duplicate ego-pose timestamp")
        self._poses = {int(row["timestamp_ns"]): transform_from_pose(row)
                       for row in poses.to_dict(orient="records")}
        self._pose_timestamps = tuple(sorted(self._poses))
        extrinsics = pd.read_feather(self.root / "calibration/egovehicle_SE3_sensor.feather")
        camera_extrinsics = extrinsics[extrinsics.sensor_name == "ring_front_center"]
        intrinsics = pd.read_feather(self.root / "calibration/intrinsics.feather")
        camera_intrinsics = intrinsics[intrinsics.sensor_name == "ring_front_center"]
        if len(camera_extrinsics) != 1 or len(camera_intrinsics) != 1:
            raise ValueError("Expected exactly one front-camera calibration")
        self.ego_T_camera = transform_from_pose(camera_extrinsics.iloc[0])
        intr = camera_intrinsics.iloc[0]
        self.K = np.array([[intr.fx_px, 0., intr.cx_px], [0., intr.fy_px, intr.cy_px], [0., 0., 1.]])
        self.camera_width, self.camera_height = int(intr.width_px), int(intr.height_px)
        missing_poses = (set(self.lidar_paths) | set(self.camera_paths)) - set(self._poses)
        if missing_poses:
            raise ValueError(f"Exact sensor-time ego pose missing for {len(missing_poses)} timestamps")

    def pose_at(self, timestamp_ns: int) -> np.ndarray:
        """Exact dataset pose only; no interpolation or extrapolation."""
        if int(timestamp_ns) not in self._poses:
            raise KeyError(f"No exact ego pose at {timestamp_ns}")
        return self._poses[int(timestamp_ns)].copy()

    def previous_camera(self, timestamp_ns: int) -> CameraFrame | None:
        index = bisect_right(self.camera_timestamps_ns, int(timestamp_ns)) - 1
        if index < 0:
            return None
        camera_time = self.camera_timestamps_ns[index]
        return CameraFrame(path=self.camera_paths[camera_time], timestamp_ns=camera_time,
                           world_T_ego_camera=self.pose_at(camera_time), ego_T_camera=self.ego_T_camera.copy(),
                           K=self.K.copy(), width=self.camera_width, height=self.camera_height)

    def ego_velocity_at(self, timestamp_ns: int, lookback_s: float = 0.1) -> np.ndarray | None:
        """Backward pose displacement over at least lookback_s, in current ego axes."""
        if lookback_s <= 0:
            raise ValueError("Positive pose velocity lookback required")
        index = bisect_right(self._pose_timestamps, int(timestamp_ns) - int(lookback_s * 1e9)) - 1
        if index < 0:
            return None
        previous_time = self._pose_timestamps[index]
        current, previous = self.pose_at(timestamp_ns), self.pose_at(previous_time)
        velocity_world = (current[:3, 3] - previous[:3, 3]) / ((int(timestamp_ns) - previous_time) * 1e-9)
        return current[:3, :3].T @ velocity_world

    def sensor_frame(self, timestamp_ns: int) -> SensorFrame:
        """Read one LiDAR sweep and sensor metadata; never read annotation files."""
        timestamp_ns = int(timestamp_ns)
        path = self.lidar_paths[timestamp_ns]
        cloud = pd.read_feather(path)
        xyz = cloud[["x", "y", "z"]].to_numpy(dtype=np.float64)
        intensity = cloud["intensity"].to_numpy(dtype=np.float64)
        if not np.isfinite(xyz).all() or not np.isfinite(intensity).all():
            raise ValueError("Nonfinite LiDAR data")
        camera = self.previous_camera(timestamp_ns)
        if camera is not None and not camera.path.is_file():
            raise FileNotFoundError(camera.path)
        return SensorFrame(log_id=self.log_id, lidar_timestamp_ns=timestamp_ns, lidar_path=path,
                           xyz_ego=xyz, intensity=intensity, world_T_ego_lidar=self.pose_at(timestamp_ns),
                           camera=camera, ego_velocity_ego_mps=self.ego_velocity_at(timestamp_ns))

    def iter_sensor_frames(self) -> Iterator[SensorFrame]:
        for timestamp_ns in self.lidar_timestamps_ns:
            yield self.sensor_frame(timestamp_ns)

    def load_reference_annotations(self) -> pd.DataFrame:
        """OFFLINE REFERENCE ONLY. Do not call this from perception/candidate code."""
        table = pd.read_feather(self.root / "annotations.feather")
        if table.duplicated(["timestamp_ns", "track_uuid"]).any():
            raise ValueError("Duplicate reference track annotation at one time")
        return table.copy()

    def load_reference_tracks_world(self) -> pd.DataFrame:
        """OFFLINE REFERENCE ONLY. Add world centers/yaw; no velocity smoothing.

        Reference velocities, if needed for scoring, must be derived separately
        with their differencing convention disclosed. They are not observations.
        """
        table = self.load_reference_annotations()
        for name in ("x_world_m", "y_world_m", "z_world_m", "yaw_world_rad"):
            table[name] = np.nan
        for timestamp_ns, group in table.groupby("timestamp_ns", sort=False):
            world_T_ego = self.pose_at(int(timestamp_ns))
            world = transform_points(world_T_ego, group[["tx_m", "ty_m", "tz_m"]].to_numpy())
            table.loc[group.index, ["x_world_m", "y_world_m", "z_world_m"]] = world
            yaw = []
            for row in group.to_dict(orient="records"):
                rotation = world_T_ego[:3, :3] @ transform_from_pose(row)[:3, :3]
                yaw.append(np.arctan2(rotation[1, 0], rotation[0, 0]))
            table.loc[group.index, "yaw_world_rad"] = yaw
        table.attrs["scope"] = "offline_ground_truth_reference_only"
        return table


def frozen_split(project_root: str | Path) -> dict:
    return json.loads((Path(project_root) / "data/manifests/av2_first6_split.json").read_text())
