"""Synthetic algebra, provenance, and common-fallback checks for final fusion."""
from copy import deepcopy
import math
import random
import unittest

from bev_risk.policy import ACTIONS, choose_action, nominal_action, stopping_distance, stress_shift
from bev_risk_v2.method import decide as legacy_decide
from bev_risk_v4.method import pair_from_state, fuse_pairs, valid_prior, decide_pair


T = 315966721660160000
EVIDENCE = dict(free_distance_m=30., current_ground_valid=True)


def state(gap=15., speed=5., age=0., timestamp=T, track=3, updated=True):
    return dict(track_id=track, gap_m=gap, lead_speed_mps=speed, age_s=age,
                last_update_ns=timestamp, last_candidate_id=f'candidate:{timestamp}',
                observations=4, initialized=True, updated=updated)


class MethodTests(unittest.TestCase):
    def test_age_stress_uses_exact_existing_transform_and_source(self):
        source = state(gap=12., speed=7., age=.7, timestamp=T-700000000, updated=False)
        original = deepcopy(source)
        pair = pair_from_state(source)
        gap, speed = stress_shift(12., 7., .7, .2)
        self.assertEqual(pair['gap_m'], gap)
        self.assertEqual(pair['support_m'], gap+speed**2/16.)
        self.assertEqual(pair['source']['last_update_ns'], T-700000000)
        self.assertTrue(pair['age_active'])
        self.assertTrue(pair['correction_nonzero'])
        self.assertNotIn('lead_speed_mps', pair)
        self.assertEqual(source, original)
        disabled = pair_from_state(source, None)
        self.assertEqual(disabled['support_m'], 12.+49./16.)
        self.assertFalse(disabled['age_active'])

    def test_valid_prior_preserves_integer_time_and_one_step_semantics(self):
        fresh = state()
        prior = state(age=.100196, timestamp=T-100196000, updated=False)
        self.assertTrue(valid_prior(fresh, prior))
        for altered in (prior | {'track_id': 4}, prior | {'initialized': False},
                        prior | {'last_update_ns': T}, prior | {'age_s': .2},
                        prior | {'last_update_ns': float(T-100196000)}):
            self.assertFalse(valid_prior(fresh, altered))
        self.assertFalse(valid_prior(fresh | {'updated': False}, prior))
        self.assertFalse(valid_prior(fresh | {'age_s': .1}, prior))
        self.assertFalse(valid_prior(None, prior))
        self.assertFalse(valid_prior(fresh, None))
        # One-step snapshot origin is separate from source age during a gap.
        old_source = state(age=.8, timestamp=T-800000000, updated=False)
        self.assertTrue(valid_prior(fresh, old_source))

    def test_envelope_is_exact_pointwise_lower_clearance(self):
        rng = random.Random(20261109)
        for _ in range(100):
            f = pair_from_state(state(rng.uniform(-5,60), rng.uniform(-3,25)))
            p = pair_from_state(state(rng.uniform(-5,60), rng.uniform(-3,25)))
            envelope = fuse_pairs(f, p)
            self.assertGreaterEqual(envelope['support_m'], envelope['gap_m'])
            for ego in (0., 4., 12., 26.):
                for action in ACTIONS:
                    e = stopping_distance(ego, action)
                    mf = min(f['gap_m'], f['support_m']-e)
                    mp = min(p['gap_m'], p['support_m']-e)
                    me = min(envelope['gap_m'], envelope['support_m']-e)
                    self.assertEqual(me, min(mf, mp))

    def test_exact_ties_and_cross_coordinate_sources_are_exposed(self):
        f = pair_from_state(state(10., 8.))  # B=14
        p = pair_from_state(state(10., 4., .1, T-100000000, updated=False))  # B=11
        result = fuse_pairs(f, p)
        self.assertEqual(result['coordinate_sources']['gap_m'], ['fresh','prior'])
        self.assertIsNone(result['coordinate_weights']['gap_m'])
        self.assertTrue(result['coordinate_ties']['gap_m'])
        self.assertEqual(result['coordinate_sources']['support_m'], ['prior'])
        f['gap_m'] = 9.
        result = fuse_pairs(f, p)
        self.assertEqual(result['coordinate_sources']['gap_m'], ['fresh'])
        self.assertEqual(result['coordinate_sources']['support_m'], ['prior'])
        self.assertNotIn('lead_speed_mps', result)
        self.assertEqual(result['sources']['prior']['last_update_ns'], T-100000000)

    def test_mean_is_fixed_equal_weight_not_optimistic_branch_choice(self):
        fresh = pair_from_state(state(10., 8.))
        prior = pair_from_state(state(12., 4., .1, T-100000000, updated=False))
        original = deepcopy((fresh, prior))
        mean = fuse_pairs(fresh, prior, 'mean')
        self.assertEqual(mean['gap_m'], 11.)
        self.assertEqual(mean['support_m'], 13.5)
        self.assertEqual(mean['coordinate_weights']['gap_m'], {'fresh':.5,'prior':.5})
        self.assertEqual((fresh, prior), original)
        self.assertEqual(fuse_pairs(fresh, None)['gap_m'], fresh['gap_m'])
        self.assertIsNone(fuse_pairs(None, prior))

    def test_conditional_error_score_bounds_all_action_margins(self):
        rng = random.Random(51)
        for _ in range(100):
            reference = pair_from_state(state(rng.uniform(-5,60), rng.uniform(-3,25)))
            estimate = pair_from_state(state(rng.uniform(-5,60), rng.uniform(-3,25)))
            q = max(0., estimate['gap_m']-reference['gap_m'],
                    estimate['support_m']-reference['support_m']) + .001
            for ego in (0.,4.,8.,12.,16.,26.):
                decision = decide_pair(estimate, ego, EVIDENCE, q)
                for row in decision['action_margins']:
                    ref_margin = min(reference['gap_m'], reference['support_m']-stopping_distance(ego,row['action']))
                    self.assertLessEqual(row['margin_m'], ref_margin+1e-12)

    def test_single_pair_policy_matches_frozen_policy_including_nominal(self):
        rng = random.Random(3)
        for _ in range(100):
            g,v,age,ego,q = rng.uniform(-5,50),rng.uniform(-3,25),rng.uniform(0,2),rng.uniform(0,32),rng.uniform(0,5)
            s = state(g,v,age)
            pair = pair_from_state(s,.2)
            new = decide_pair(pair,ego,EVIDENCE,q)
            old = choose_action(g,v,ego,age,.2,q)
            self.assertEqual(new['action'],old['action'])
            self.assertEqual(new['nominal_action'],old['nominal_action'])
            self.assertEqual(new['reason'],old['reason'])
            self.assertLessEqual(new['action'],nominal_action(ego))

    def test_same_buffer_envelope_cannot_authorize_more_than_fresh(self):
        rng = random.Random(4)
        for _ in range(100):
            f = pair_from_state(state(rng.uniform(-5,60),rng.uniform(0,25)))
            p = pair_from_state(state(rng.uniform(-5,60),rng.uniform(0,25)))
            ego,q = rng.uniform(0,32),rng.uniform(0,4)
            self.assertLessEqual(decide_pair(fuse_pairs(f,p),ego,EVIDENCE,q)['action'],
                                 decide_pair(f,ego,EVIDENCE,q)['action'])

    def test_missing_evidence_uses_common_zero_buffer_fallback(self):
        for evidence in (EVIDENCE, dict(free_distance_m=0.,current_ground_valid=False),
                         dict(free_distance_m=12.,current_ground_valid=True)):
            for ego in (0.,4.,12.):
                expected=legacy_decide(None,ego,evidence,buffer_m=0.,repair=True)
                actual=decide_pair(None,ego,evidence,buffer_m=1000.)
                for key,value in expected.items():self.assertEqual(actual[key],value)
                self.assertFalse(actual['calibration_applied'])
                self.assertEqual(actual['buffer_m'],0.)
        self.assertIsNone(pair_from_state(None))
        self.assertIsNone(pair_from_state(state() | {'initialized':False}))

    def test_zero_margin_is_not_admitted_and_invalid_inputs_fail(self):
        pair=pair_from_state(state(0.,0.))
        self.assertEqual(decide_pair(pair,0.,EVIDENCE)['reason'],'full_brake_fallback')
        for invalid in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):decide_pair(pair,4.,EVIDENCE,invalid)
        with self.assertRaises(ValueError):fuse_pairs(pair,pair,'best_action')
        with self.assertRaises(ValueError):pair_from_state(state(age=-1))
        with self.assertRaises(ValueError):pair_from_state(state(timestamp=float(T)))


if __name__ == '__main__':
    unittest.main()
