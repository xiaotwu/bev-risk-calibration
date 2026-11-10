"""Synthetic geometry/causality checks; not empirical evaluation samples."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bev_risk.perception import (CameraDetector, associate_camera, estimate_ground,
                                 lidar_clusters, perceive_frame, project_to_camera)


def synthetic_cloud():
    x, y = np.meshgrid(np.arange(2., 40., .5), np.arange(-8., 8., .5))
    ground = np.column_stack([x.ravel(), y.ravel(), .04*x.ravel()+.02*y.ravel()-1.5])
    objects = []
    for start in (10., 25.):
        x, y, h = np.meshgrid(np.arange(start, start+3., .3), np.arange(-.9, 1., .3), [.5, 1., 1.5])
        objects.append(np.column_stack([x.ravel(), y.ravel(), .04*x.ravel()+.02*y.ravel()-1.5+h.ravel()]))
    return np.vstack([ground, *objects])


def frame(points=None):
    camera_pose = np.eye(4)
    # Optical x right, y down, z forward -> ego x forward, y left, z up.
    camera_pose[:3, :3] = [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]
    world = np.eye(4)
    world[0, 3] = 1.
    return SimpleNamespace(log_id="unit-log", lidar_timestamp_ns=100_000_000,
        lidar_path=Path("unit-lidar.feather"), xyz_ego=points, world_T_ego_lidar=world,
        camera=SimpleNamespace(path=Path("unit-camera.jpg"), timestamp_ns=50_000_000,
                               world_T_ego_camera=np.eye(4), ego_T_camera=camera_pose,
                               K=np.array([[100., 0., 100.], [0., 100., 100.], [0., 0., 1.]]),
                               width=200, height=200))


class PerceptionTests(unittest.TestCase):
    def test_ground_fit_preserves_two_separate_vehicles(self):
        points = synthetic_cloud()
        plane, quality = estimate_ground(points)
        np.testing.assert_allclose(plane, [.04, .02, -1.5], atol=1e-8)
        self.assertGreater(quality, .95)
        candidates, diagnostics = lidar_clusters(points)
        self.assertEqual(len(candidates), 2)
        self.assertGreater(diagnostics["n_nonground_points"], 100)
        self.assertTrue(all(c["geometry_quality"] > .9 for c in candidates))
        self.assertAlmostEqual(candidates[0]["observed_rear_x_ego_m"], 10.)

    def test_projection_compensates_actual_ego_motion(self):
        uv, valid = project_to_camera(np.array([[9., 1., 0.], [-2., 0., 0.]]), frame())
        np.testing.assert_allclose(uv[0], [90., 100.])
        self.assertEqual(valid.tolist(), [True, False])

    def test_future_camera_rejected_and_old_camera_unused(self):
        f = frame(np.empty((0, 3)))
        f.camera.timestamp_ns = 100_000_001
        with self.assertRaisesRegex(ValueError, "Future"):
            perceive_frame(f, None)
        f.camera.timestamp_ns = -200_000_000
        _, valid = project_to_camera(np.array([[9., 0., 0.]]), f)
        self.assertFalse(valid.any())

    def test_camera_association_one_to_one_without_dropping_lidar(self):
        points = np.array([[9., -1., -1.], [9., 1., -1.], [9., -1., 1.], [9., 1., 1.]])
        candidates = [dict(points_ego=points.copy()), dict(points_ego=points.copy()+[1., 0., 0.])]
        detection = dict(box_xyxy=[80., 80., 120., 120.], score=.8, class_name="car")
        associate_camera(candidates, [detection], frame())
        self.assertEqual(len(candidates), 2)
        self.assertEqual(sum(c["camera_score"] > 0 for c in candidates), 1)

    def test_no_camera_keeps_lidar_world_geometry_and_sources(self):
        f = frame(synthetic_cloud())
        f.camera = None
        candidates, _ = perceive_frame(f, None)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(c["camera_score"] == 0. for c in candidates))
        np.testing.assert_allclose(np.array(candidates[0]["center_world"])-candidates[0]["center_ego"], [1., 0., 0.])
        self.assertEqual(candidates[0]["lidar_timestamp_ns"], 100_000_000)
        self.assertNotIn("points_ego", candidates[0])

    def test_total_cache_budget_counts_multiple_files(self):
        with TemporaryDirectory() as directory:
            detector = CameraDetector.__new__(CameraDetector)
            detector.root = Path(directory)
            (detector.root / "a").write_bytes(b"12")
            (detector.root / "b").write_bytes(b"34")
            with patch("bev_risk.perception.CAMERA_CACHE_BUDGET", 5):
                detector._budget(1)
                with self.assertRaisesRegex(RuntimeError, "total-size"):
                    detector._budget(2)


if __name__ == "__main__":
    unittest.main()
