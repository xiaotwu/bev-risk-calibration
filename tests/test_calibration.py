"""Bounded synthetic checks for training-only whole-log calibration."""
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_v4.calibration import (
    ESTIMATORS, fit_folds, joint_optimism_error, weighted_inverse_cdf,
)
from bev_risk.policy import choose_action, stopping_distance


def rows_for(log, values, timestamp_start=100):
    return [dict(log_id=log, timestamp_ns=timestamp_start+i, estimator=est,
                 error_m=float(value), wrong_lead=False, elapsed_s=2.+i*.1,
                 reference_available=True, reference_has_lead=True,
                 reference_velocity_valid=True, selected_state=True)
            for est in ESTIMATORS for i, value in enumerate(values)]


class CalibrationTests(unittest.TestCase):
    def test_joint_score_requires_both_gap_and_support(self):
        self.assertEqual(joint_optimism_error(3, 100, 1, 99), 2.)
        self.assertEqual(joint_optimism_error(1, 104, 2, 100), 4.)
        self.assertEqual(joint_optimism_error(1, 5, 2, 7), -1.)
        with self.assertRaises(ValueError):
            joint_optimism_error(1, float('nan'), 2, 3)

    def test_inverse_cdf_exact_no_interpolation(self):
        self.assertEqual(weighted_inverse_cdf([1, 2, 3, 4], [1]*4, .5), 2.)
        self.assertEqual(weighted_inverse_cdf([10, 0, 5], [Fraction(1, 3)]*3, 1.), 10.)
        self.assertEqual(weighted_inverse_cdf([2, 1], [Fraction(1, 10), Fraction(9, 10)], .9), 1.)
        self.assertEqual(weighted_inverse_cdf([2, 1], [Fraction(1, 10), Fraction(9, 10)], .91), 2.)
        self.assertEqual(weighted_inverse_cdf([1, 1, 2], [1, 1, 1], .5), 1.)

    def test_unequal_counts_do_not_dominate_equal_log_weights(self):
        rows = rows_for('many', [0.]*100) + rows_for('rare', [10.]) + rows_for('held', [500.])
        fold = fit_folds(rows, ['many', 'rare', 'held'])['held']
        self.assertEqual(fold['q_m'], {est: 10. for est in ESTIMATORS})
        for counts in fold['counts'].values():
            self.assertEqual(counts['training_rows'], 101)
            self.assertEqual(counts['effective_log_count'], 2)
            self.assertEqual(counts['rows_by_log'], {'many': 100, 'rare': 1})

    def test_held_out_values_and_counts_do_not_change_own_artifact_or_decision(self):
        logs = ['alpha', 'beta', 'gamma']
        original = rows_for('alpha', [1., 2.]) + rows_for('beta', [.1, .2]) + rows_for('gamma', [.3, .4])
        mutated = deepcopy(original)
        for row in mutated:
            if row['log_id'] == 'alpha':
                row['error_m'] = 1e8
                row['wrong_lead'] = True
        mutated += rows_for('alpha', [1e9]*13, timestamp_start=1000)
        before = fit_folds(original, logs)
        after = fit_folds(mutated, logs)
        self.assertEqual(before['alpha'], after['alpha'])
        # This is an actual frozen policy call, with one reserve application.
        for estimator in ESTIMATORS:
            q_before = before['alpha']['q_m'][estimator]
            q_after = after['alpha']['q_m'][estimator]
            d_before = choose_action(14., 0., 10., buffer_m=q_before)
            d_after = choose_action(14., 0., 10., buffer_m=q_after)
            self.assertEqual(d_before, d_after)
        self.assertNotEqual(before['beta'], after['beta'])

    def test_empty_logs_remain_visible_without_zero_error_samples(self):
        logs = ['a', 'b', 'empty']
        rows = rows_for('a', [1.]) + rows_for('b', [9.])
        fold = fit_folds(rows, logs)['a']
        self.assertEqual(fold['training_log_ids'], ['b', 'empty'])
        self.assertEqual(fold['q_m']['fresh'], 9.)
        counts = fold['counts']['fresh']
        self.assertEqual(counts['empty_training_log_ids'], ['empty'])
        self.assertEqual(counts['rows_by_log'], {'b': 1, 'empty': 0})
        self.assertEqual(counts['training_rows'], 1)
        self.assertEqual(counts['effective_log_count'], 1)

    def test_whole_empty_training_pool_is_error(self):
        with self.assertRaisesRegex(ValueError, 'No valid calibration rows'):
            fit_folds(rows_for('only', [1.]), ['only', 'empty'])
        rows = rows_for('a', [1.]) + [r for r in rows_for('b', [1.]) if r['estimator'] == 'fresh']
        with self.assertRaisesRegex(ValueError, "'mean'"):
            fit_folds(rows, ['a', 'b'])

    def test_wrong_selected_lead_is_retained_in_fit(self):
        rows = rows_for('held', [.1]) + rows_for('correct', [.1]) + rows_for('wrong', [20.])
        for row in rows:
            if row['log_id'] == 'wrong':
                row['wrong_lead'] = True
        fold = fit_folds(rows, ['held', 'correct', 'wrong'])['held']
        self.assertEqual(fold['q_m']['fresh'], 20.)
        self.assertEqual(fold['counts']['fresh']['wrong_lead_rows'], 1)
        self.assertEqual(fold['counts']['fresh']['wrong_lead_rows_by_log'], {'correct': 0, 'wrong': 1})

    def test_negative_scores_preserved_and_reserve_clipped_once_at_zero(self):
        fold = fit_folds(rows_for('a', [-2., -1.]) + rows_for('b', [-3.]), ['a', 'b'])['b']
        self.assertEqual(fold['q_m']['envelope'], 0.)
        self.assertEqual(fold['unclipped_quantiles_m']['envelope'], -1.)
        self.assertEqual(fold['counts']['envelope']['negative_error_rows'], 2)

    def test_order_invariance_and_roster_diagnostics(self):
        rows = rows_for('a', [1., 4., 2.]) + rows_for('b', [.3, .2]) + rows_for('c', [.7])
        self.assertEqual(fit_folds(rows, ['a', 'b', 'c']), fit_folds(list(reversed(rows)), ['a', 'b', 'c']))
        with self.assertRaisesRegex(ValueError, 'unique'):
            fit_folds(rows, ['a', 'a', 'b'])
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            fit_folds(rows, ['a', 'b'])

    def test_invalid_rows_and_duplicate_timestamps_are_rejected(self):
        base = rows_for('a', [1.]) + rows_for('b', [2.])
        for bad in (None, float('nan'), float('inf'), True):
            rows = deepcopy(base); rows[0]['error_m'] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                fit_folds(rows, ['a', 'b'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            fit_folds(base+[deepcopy(base[0])], ['a', 'b'])
        rows = deepcopy(base); rows[0]['timestamp_ns'] = 100.0
        with self.assertRaisesRegex(ValueError, 'integer'):
            fit_folds(rows, ['a', 'b'])

    def test_warmup_reference_eligibility_and_fixed_level_are_enforced(self):
        base = rows_for('a', [1.]) + rows_for('b', [2.])
        rows = deepcopy(base); rows[0]['elapsed_s'] = 1.999999999
        with self.assertRaisesRegex(ValueError, 'pre-warm-up'):
            fit_folds(rows, ['a', 'b'])
        for flag in ('reference_available', 'reference_has_lead', 'reference_velocity_valid', 'selected_state'):
            rows = deepcopy(base); rows[0][flag] = False
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'not eligible'):
                fit_folds(rows, ['a', 'b'])
        with self.assertRaisesRegex(ValueError, '0.90'):
            fit_folds(base, ['a', 'b'], level=.95)

    def test_joint_score_bound_protects_both_contract_inequalities_conditionally(self):
        fixtures = [(1., 30., -1., 30.), (14., 16., 13.7, 15.5),
                    (10., 30., 9., 25.), (30., 35., 31., 36.)]
        for g_est, B_est, g_ref, B_ref in fixtures:
            q = max(0., joint_optimism_error(g_est, B_est, g_ref, B_ref))
            g_lower, B_lower = g_est-q, B_est-q
            self.assertLessEqual(g_lower, g_ref+1e-12)
            self.assertLessEqual(B_lower, B_ref+1e-12)
            for action in (-6., -3., 0., 1.5, 3.):
                E = stopping_distance(10., action)
                if g_lower > 0. and B_lower > E:
                    self.assertGreater(g_ref, 0.)
                    self.assertGreater(B_ref, E)
        # This algebraic test does not assert that an empirical quantile covers
        # a future sample, or that the unconditional full-brake fallback is safe.


if __name__ == '__main__':
    unittest.main()
