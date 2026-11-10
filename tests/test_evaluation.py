"""Synthetic causality and endpoint checks; no held-out data is loaded."""
import unittest

import numpy as np

from bev_risk.evaluation import ReferenceLog, score
from bev_risk.geometry import corridor_gap
from bev_risk.policy import nominal_action


CONFIG = dict(ego_front_offset_m=2.5, corridor_half_width_m=1.75, range_m=60.)


def pose(t, x=0., y=0., yaw=0.):
    return dict(timestamp_ns=t, qw=np.cos(yaw/2), qx=0., qy=0., qz=np.sin(yaw/2),
                tx_m=x, ty_m=y, tz_m=0.)


def box(t, identity="lead", x=16.5, y=0., category="REGULAR_VEHICLE", yaw=0.):
    return dict(pose(t, x, y, yaw), track_uuid=identity, category=category,
                length_m=4., width_m=2., height_m=2.)


def state(x=16.5, y=0., gap=12., speed=0.):
    return dict(track_id=7, center_world=[x, y, 0.], gap_m=gap,
                lead_speed_mps=speed, age_s=0., initialized=True, updated=True)


def reference(records=None):
    records = records if records is not None else [box(0), box(200_000_000)]
    times = sorted({row["timestamp_ns"] for row in records})
    log = ReferenceLog.from_tables(records, [pose(t) for t in times], CONFIG)
    return log.at(200_000_000, np.eye(4), 10.)


class EvaluationTests(unittest.TestCase):
    def test_backward_velocity_uses_world_motion_not_ego_relative_change(self):
        records = [box(0, x=20.), box(200_000_000, x=20.), box(400_000_000, x=999.)]
        log = ReferenceLog.from_tables(records,
                                      [pose(0), pose(200_000_000, x=2.), pose(400_000_000, x=4.)], CONFIG)
        current = np.eye(4)
        current[0, 3] = 2.
        ref = log.at(200_000_000, current, 10.)
        self.assertAlmostEqual(ref["lead"]["lead_speed_mps"], 10.)
        self.assertEqual(ref["lead"]["velocity_previous_timestamp_ns"], 0)
        self.assertAlmostEqual(ref["lead"]["gap_m"], 15.5)
        scored = score(ref, None, -6., 10., CONFIG)
        self.assertAlmostEqual(scored["reference_gap_m"], 15.5)
        self.assertAlmostEqual(scored["reference_lead_speed_mps"], 10.)
        self.assertAlmostEqual(scored["reference_velocity_lookback_s"], .2)

    def test_missing_or_too_old_velocity_is_not_zero(self):
        log = ReferenceLog.from_tables([box(0), box(600_000_000)], [pose(0), pose(600_000_000)], CONFIG)
        first = log.at(0, np.eye(4), 10.)
        stale = log.at(600_000_000, np.eye(4), 10.)
        for ref in (first, stale):
            self.assertIsNone(ref["lead"]["lead_speed_mps"])
            self.assertFalse(ref["lead"]["velocity_valid"])
            scored = score(ref, state(), 3., 10., CONFIG)
            self.assertFalse(scored["false_admission_eligible"])
            self.assertIsNone(scored["reference_action_mps2"])
            self.assertIsNone(scored["reference_lead_speed_mps"])
            self.assertIsNotNone(scored["reference_gap_m"])

    def test_material_false_admission_uses_shared_reference_not_identity_match(self):
        ref = reference()
        scored = score(ref, state(x=100., gap=95.), 3., 10., CONFIG)
        self.assertTrue(scored["reference_feasible"])
        self.assertTrue(scored["false_admission_eligible"])
        self.assertTrue(scored["false_admission"])
        self.assertGreater(scored["reference_deficit_m"], .1)
        self.assertTrue(scored["unmatched_estimated_lead"])
        self.assertIsNone(scored["support_signed_error_m"])

    def test_infeasible_full_brake_is_separate_from_false_admission(self):
        ref = reference([box(0, x=5.5), box(200_000_000, x=5.5)])
        scored = score(ref, state(x=5.5, gap=1.), 3., 10., CONFIG)
        self.assertFalse(scored["reference_feasible"])
        self.assertFalse(scored["false_admission_eligible"])
        self.assertIsNone(scored["false_admission"])
        self.assertLess(scored["reference_full_brake_margin_m"], -.1)
        self.assertGreater(scored["extra_deficit_over_full_brake_m"], 0.)

    def test_reference_empty_has_nominal_action_but_no_false_admission_denominator(self):
        ref = reference([box(0, category="PEDESTRIAN"), box(200_000_000, category="PEDESTRIAN")])
        scored = score(ref, None, -6., 10., CONFIG)
        self.assertTrue(scored["reference_empty"])
        self.assertEqual(scored["reference_action_mps2"], nominal_action(10.))
        self.assertTrue(scored["overconservative_action"])
        self.assertFalse(scored["false_admission_eligible"])
        self.assertIsNone(scored["false_admission"])

    def test_missing_annotation_timestamp_is_not_reference_empty(self):
        log = ReferenceLog.from_tables([box(0), box(400_000_000)],
                                      [pose(0), pose(200_000_000), pose(400_000_000)], CONFIG)
        ref = log.at(200_000_000, np.eye(4), 10.)
        scored = score(ref, None, -6., 10., CONFIG)
        self.assertFalse(scored["reference_available"])
        self.assertFalse(scored["reference_empty"])
        self.assertIsNone(scored["overconservative_action"])

    def test_pedestrian_braking_is_not_called_unmatched_to_all_annotations(self):
        ref = reference([box(0, category="PEDESTRIAN"), box(200_000_000, category="PEDESTRIAN")])
        scored = score(ref, state(), -6., 10., CONFIG)
        self.assertTrue(scored["reference_empty"])
        self.assertTrue(scored["unmatched_estimated_lead"])
        self.assertEqual(scored["matched_any_gt_category"], "PEDESTRIAN")
        self.assertTrue(scored["out_of_scope_object_match"])
        self.assertFalse(scored["unmatched_to_any_annotation"])
        self.assertTrue(scored["overconservative_action"])
        self.assertFalse(scored["false_admission_eligible"])

    def test_all_class_matching_does_not_change_vehicle_reference(self):
        records = [box(t, identity="vehicle", x=20.) for t in (0, 200_000_000)]
        records += [box(t, identity="pedestrian", x=10., category="PEDESTRIAN") for t in (0, 200_000_000)]
        ref = reference(records)
        scored = score(ref, state(x=10., gap=5.5), -6., 10., CONFIG)
        self.assertEqual(ref["lead"]["track_uuid"], "vehicle")
        self.assertEqual(scored["reference_track_uuid"], "vehicle")
        self.assertEqual(scored["matched_any_gt_category"], "PEDESTRIAN")
        self.assertTrue(scored["out_of_scope_object_match"])
        self.assertTrue(scored["false_admission_eligible"])

    def test_wrong_annotated_lead_does_not_supply_support_error(self):
        records = [box(t, identity="near", x=16.5) for t in (0, 200_000_000)]
        records += [box(t, identity="far", x=40.) for t in (0, 200_000_000)]
        scored = score(reference(records), state(x=40., gap=35.5), 3., 10., CONFIG)
        self.assertTrue(scored["wrong_annotated_lead"])
        self.assertEqual(scored["matched_gt_track_uuid"], "far")
        self.assertTrue(scored["false_admission"])
        self.assertIsNone(scored["support_signed_error_m"])

    def test_footprint_matching_has_explicit_point_to_polygon_tolerance(self):
        ref = reference()
        accepted = score(ref, state(y=1.75), -6., 10., CONFIG)
        rejected = score(ref, state(y=1.751), -6., 10., CONFIG)
        self.assertTrue(accepted["matched_reference_lead"])
        self.assertAlmostEqual(accepted["match_footprint_distance_m"], .75)
        self.assertTrue(rejected["unmatched_estimated_lead"])

    def test_overlapping_reference_gap_is_retained(self):
        ref = reference([box(0, x=4.), box(200_000_000, x=4.)])
        self.assertTrue(ref["has_lead"])
        self.assertAlmostEqual(ref["lead"]["gap_m"], -.5)
        self.assertFalse(score(ref, None, -6., 10., CONFIG)["reference_feasible"])

    def test_slanted_polygon_gap_clips_lateral_corridor_first(self):
        diamond = np.array([[10., 3.], [13., 0.], [16., 3.], [13., 6.]])
        gap = corridor_gap(diamond, CONFIG)
        self.assertAlmostEqual(gap, 11.25-2.5)
        self.assertGreater(gap, diamond[:, 0].min()-2.5)

    def test_never_initialized_track_is_not_removed_from_reference_scoring(self):
        scored = score(reference(), None, -6., 10., CONFIG)
        self.assertTrue(scored["no_estimated_lead"])
        self.assertTrue(scored["false_admission_eligible"])
        self.assertFalse(scored["false_admission"])
        self.assertTrue(scored["overconservative_action"])


if __name__ == "__main__":
    unittest.main()
