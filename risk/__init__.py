"""Portable CV tracking and stale-observation braking-risk calculations."""

import json
from pathlib import Path

from .policy import MotionPolicy, stress_shift
from .tracker import LegacyCVTracker

__all__ = ["LegacyCVTracker", "MotionPolicy", "load_calibration", "stress_shift"]


def load_calibration():
    """Load the frozen age-bin factors for the CV reference and correction."""
    with Path(__file__).with_name("calibration.json").open(encoding="utf-8") as stream:
        calibration = json.load(stream)
    return {calibration["tracker"]: calibration}
