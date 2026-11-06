"""Regression checks for the portable risk and tracker interfaces."""

import unittest

import numpy as np

from . import LegacyCVTracker, MotionPolicy, load_calibration, stress_shift
from .control import ACTIONS, displacement
from .tail_integral import adaptive_reference, conditional_contact_probability


class RiskTests(unittest.TestCase):
    def test_stop_and_no_reverse(self):
        for velocity in (-3.0, 0.0, 1.5, 8.0, 18.0):
            for age in (0.0, 1.0, 1.25, 2.0, 5.0, 8.0):
                horizon, gap, speed = stress_shift(velocity, age)
                clipped = max(0.0, velocity)
                self.assertAlmostEqual(gap, clipped * horizon - displacement(clipped, -3.0, horizon))
                self.assertAlmostEqual(speed, clipped - max(0.0, clipped - 3.0 * horizon))
        self.assertEqual(stress_shift(12.0, 2.0), (1.0, 1.5, 3.0))

    def test_tail_against_adaptive_quadrature(self):
        for action in ACTIONS:
            args = (17.0, 8.0, 2.0, 1.0, 0.7, 16.0, -3.0, action)
            online = conditional_contact_probability(*args)
            self.assertAlmostEqual(online, adaptive_reference(*args)["probability"], places=9)

    def test_undelivered_duplicate_and_negative_packets(self):
        tracker = LegacyCVTracker()
        front = {"center": [20.0, 0.0, 0.0], "velocity": [10.0, 0.0, 0.0], "score": 0.8}
        positive = {"frame": 1, "acquisition_time": 0.0, "front": front}
        row = tracker.update(positive, 0.0, 12.0, 0.0, 0.2)
        self.assertTrue(row["measurement_used"])
        row = tracker.update(positive, 2.4, 12.0, 0.0, 0.4)
        self.assertFalse(row["measurement_used"])
        row = tracker.update(None, 4.8, 12.0, 0.0, 0.6)
        self.assertFalse(row["packet_received"])
        negative = {"frame": 2, "acquisition_time": 0.7, "front": None}
        row = tracker.update(negative, 7.2, 12.0, 0.0, 0.8)
        self.assertEqual(row["tracker_diagnostics"]["last_detection_acquisition_s"], 0.0)
        self.assertLess(row["presence"], 0.99)

    def test_policy_limits_and_fresh_parity(self):
        policy = {"tracker": "cv_current", "param": 0.01}
        calibration = load_calibration()
        reference = MotionPolicy(policy, calibration)
        corrected = MotionPolicy(dict(policy, age_stress=True), calibration)
        self.assertEqual(calibration["cv_current"]["n_scene_histories"], 256)
        for positive in (False, True):
            for age in (0.8, 1.0, 1.4, 3.8):
                for speed in (-2.0, 0.5, 12.0, 20.0):
                    feature = np.zeros(15)
                    feature[2] = age
                    feature[14] = positive
                    tracker = dict(feature=feature, presence=0.8, mean_gap=24.0,
                                   mean_vlead=speed, base_sigma_gap=2.0, base_sigma_v=2.0,
                                   covariance=[[4.0, 2.0], [2.0, 4.0]], age_s=age)
                    tracker["legacy_reference"] = tracker.copy()
                    tracker["tracker_diagnostics"] = {
                        "last_detection_acquisition_s": 4.0 - age if positive else None
                    }
                    cv = reference.step(tracker, 16.0, -3.0, 4.0)
                    stress = corrected.step(tracker, 16.0, -3.0, 4.0)
                    a = cv["diagnostics"]["candidates"]["risk"]
                    b = stress["diagnostics"]["candidates"]["risk"]
                    self.assertTrue(np.all(b >= a - 1e-9))
                    self.assertLessEqual(stress["command"], cv["command"])
                    if not positive or age <= 1.0 or speed <= 0:
                        np.testing.assert_array_equal(a, b)
                        self.assertEqual(stress["command"], cv["command"])


if __name__ == "__main__":
    unittest.main()
