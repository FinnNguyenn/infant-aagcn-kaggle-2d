import unittest

import numpy as np

from modules.corruption_audit import compare_skeletons, motion_energy, velocity_correlation


class CorruptionAuditTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(8, dtype=float)
        self.j = np.zeros((8, 3, 2), dtype=float)
        self.j[:, 0, 0] = time
        self.j[:, 1, 0] = time + 1.0
        self.j[:, 2, 1] = 0.5 * time
        self.b = np.zeros_like(self.j)
        self.b[:, 1] = self.j[:, 1] - self.j[:, 0]
        self.b[:, 2] = self.j[:, 2] - self.j[:, 0]

    def test_identity_metrics(self):
        metrics = compare_skeletons(
            self.j, self.j.copy(), self.b, self.b.copy(), body_scale=1.0
        )
        self.assertAlmostEqual(metrics["j_motion_energy_ratio"], 1.0)
        self.assertAlmostEqual(metrics["j_velocity_correlation"], 1.0)
        self.assertAlmostEqual(metrics["bone_length_relative_error_p95"], 0.0)
        self.assertAlmostEqual(metrics["displacement_p95_body_scale"], 0.0)

    def test_scale_changes_energy_and_bone_lengths_predictably(self):
        factor = 1.1
        changed_j = self.j * factor
        changed_b = self.b * factor
        metrics = compare_skeletons(
            self.j, changed_j, self.b, changed_b, body_scale=1.0
        )
        self.assertAlmostEqual(metrics["j_motion_energy_ratio"], factor**2)
        self.assertAlmostEqual(metrics["j_velocity_correlation"], 1.0)
        self.assertAlmostEqual(metrics["bone_length_relative_error_mean"], 0.1)

    def test_masked_reconstruction_metrics(self):
        changed = self.j.copy()
        mask = np.zeros(self.j.shape[:2], dtype=bool)
        mask[2, 1] = True
        changed[2, 1, 0] += 0.2
        changed_b = self.b.copy()
        metrics = compare_skeletons(
            self.j,
            changed,
            self.b,
            changed_b,
            body_scale=2.0,
            corruption_mask=mask,
        )
        self.assertEqual(metrics["reconstruction_masked_count"], 1)
        self.assertAlmostEqual(
            metrics["reconstruction_displacement_p95_body_scale"], 0.1
        )

    def test_degenerate_velocity_is_reported(self):
        static = np.zeros((4, 2, 2), dtype=float)
        self.assertEqual(motion_energy(static), 0.0)
        self.assertEqual(velocity_correlation(static, static.copy()), 1.0)


if __name__ == "__main__":
    unittest.main()
