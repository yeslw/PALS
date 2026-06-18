import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_dependency_budget_ablation import build_layerwise_sparsity_schedule


class ScheduleGeneratorDebugTests(unittest.TestCase):
    def test_strictly_increasing_scores_produce_monotonic_schedule_without_anomalous_single_point_floor(self):
        scores = np.arange(1, 33, dtype=np.float64)
        assigned, score_norm, lower_bound, upper_bound, debug_payload = build_layerwise_sparsity_schedule(
            scores,
            global_sparsity=0.7,
            alpha=0.15,
            return_debug_payload=True,
        )

        self.assertTrue(np.all(np.diff(assigned) <= 1e-12))
        self.assertAlmostEqual(float(np.min(assigned)), lower_bound, places=12)
        self.assertAlmostEqual(float(np.max(assigned)), upper_bound, places=12)
        self.assertEqual(int(np.count_nonzero(np.isclose(assigned, lower_bound, rtol=0.0, atol=1e-12))), 1)
        self.assertEqual(int(np.argmax(scores)), int(np.argmin(assigned)))
        trace = debug_payload["target_value_trace"]
        self.assertIsNotNone(trace)
        self.assertEqual(trace["layer_idx"], 31)
        self.assertEqual(trace["stage"], "mapped_sparsity_before_clip")
        self.assertIn("lower_bound", trace["rule"])
        np.testing.assert_allclose(score_norm, np.linspace(0.0, 1.0, 32, dtype=np.float64), atol=1e-12, rtol=0.0)

    def test_all_equal_scores_produce_uniform_schedule(self):
        scores = np.ones(32, dtype=np.float64)
        assigned, score_norm, lower_bound, upper_bound, debug_payload = build_layerwise_sparsity_schedule(
            scores,
            global_sparsity=0.7,
            alpha=0.15,
            return_debug_payload=True,
        )

        expected = np.full(32, 0.7, dtype=np.float64)
        np.testing.assert_allclose(score_norm, np.full(32, 0.5, dtype=np.float64), atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(assigned, expected, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(debug_payload["mapped_sparsity_before_clip"], expected, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(debug_payload["mapped_sparsity_after_clip"], expected, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(debug_payload["mapped_sparsity_after_renorm"], expected, atol=1e-12, rtol=0.0)
        self.assertFalse(debug_payload["plateau_warning"])
        self.assertAlmostEqual(lower_bound, 0.55, places=12)
        self.assertAlmostEqual(upper_bound, 0.85, places=12)

    def test_permutation_consistency_after_restoring_original_layer_order(self):
        scores = np.asarray(
            [0.4, 1.2, 3.5, 0.7, 2.6, 4.9, 1.8, 0.9, 2.1, 5.4, 3.2, 4.1, 6.3, 7.7, 8.8, 9.9,
             10.1, 11.4, 12.6, 13.2, 14.8, 15.9, 16.3, 17.5, 18.7, 19.1, 20.4, 21.6, 22.8, 23.3, 24.4, 25.5],
            dtype=np.float64,
        )
        permutation = np.asarray(
            [7, 2, 15, 0, 28, 4, 19, 10, 31, 6, 23, 12, 1, 26, 17, 9, 14, 30, 5, 21, 8, 18, 3, 24, 11, 29, 13, 20, 27, 16, 22, 25],
            dtype=np.int64,
        )
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(permutation.size, dtype=np.int64)

        assigned_ref, _, _, _, debug_ref = build_layerwise_sparsity_schedule(
            scores,
            global_sparsity=0.7,
            alpha=0.15,
            return_debug_payload=True,
        )
        assigned_perm, _, _, _, debug_perm = build_layerwise_sparsity_schedule(
            scores[permutation],
            global_sparsity=0.7,
            alpha=0.15,
            return_debug_payload=True,
        )

        np.testing.assert_allclose(assigned_perm[inverse], assigned_ref, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(debug_perm["normalized_score"][inverse], debug_ref["normalized_score"], atol=1e-12, rtol=0.0)
        np.testing.assert_array_equal(debug_perm["rank"][inverse], debug_ref["rank"])

    def test_rank_generators_depend_only_on_rank_order_not_score_spacing(self):
        scores_a = np.linspace(-3.0, 4.0, 32, dtype=np.float64)
        scores_b = np.exp(scores_a)
        param_counts = np.asarray([1000 + 17 * idx for idx in range(32)], dtype=np.float64)

        for generator_mode in ("rank_linear", "rank_logistic"):
            assigned_a, _, _, _, debug_a = build_layerwise_sparsity_schedule(
                scores_a,
                global_sparsity=0.7,
                alpha=0.15,
                param_counts=param_counts,
                generator_mode=generator_mode,
                profile_temperature=0.15,
                return_debug_payload=True,
            )
            assigned_b, _, _, _, debug_b = build_layerwise_sparsity_schedule(
                scores_b,
                global_sparsity=0.7,
                alpha=0.15,
                param_counts=param_counts,
                generator_mode=generator_mode,
                profile_temperature=0.15,
                return_debug_payload=True,
            )

            np.testing.assert_array_equal(debug_a["rank"], debug_b["rank"])
            np.testing.assert_allclose(debug_a["percentile"], debug_b["percentile"], atol=1e-12, rtol=0.0)
            np.testing.assert_allclose(debug_a["template_value"], debug_b["template_value"], atol=1e-12, rtol=0.0)
            np.testing.assert_allclose(assigned_a, assigned_b, atol=1e-12, rtol=0.0)

    def test_rank_generators_satisfy_bounds_and_weighted_mean(self):
        scores = np.asarray(
            [0.4, 1.2, 3.5, 0.7, 2.6, 4.9, 1.8, 0.9, 2.1, 5.4, 3.2, 4.1, 6.3, 7.7, 8.8, 9.9,
             10.1, 11.4, 12.6, 13.2, 14.8, 15.9, 16.3, 17.5, 18.7, 19.1, 20.4, 21.6, 22.8, 23.3, 24.4, 25.5],
            dtype=np.float64,
        )
        param_counts = np.asarray([500 + 29 * idx for idx in range(scores.size)], dtype=np.float64)

        for generator_mode in ("rank_linear", "rank_logistic"):
            assigned, _, lower_bound, upper_bound, debug_payload = build_layerwise_sparsity_schedule(
                scores,
                global_sparsity=0.7,
                alpha=0.15,
                param_counts=param_counts,
                generator_mode=generator_mode,
                profile_temperature=0.15,
                return_debug_payload=True,
            )

            self.assertTrue(np.all(assigned >= lower_bound - 1e-12))
            self.assertTrue(np.all(assigned <= upper_bound + 1e-12))
            weighted_mean = float(np.sum(assigned * param_counts) / np.sum(param_counts))
            self.assertAlmostEqual(weighted_mean, 0.7, places=12)
            self.assertAlmostEqual(float(debug_payload["weighted_mean"]), 0.7, places=12)
            self.assertAlmostEqual(float(debug_payload["weighted_mean_error"]), 0.0, places=10)
            self.assertEqual(debug_payload["generator_mode"], generator_mode)


if __name__ == "__main__":
    unittest.main()
