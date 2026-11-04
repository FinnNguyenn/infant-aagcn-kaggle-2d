import unittest

import numpy as np

from modules.constants import NODE_INDEX
from modules.corruption import (
    CorruptionConfig,
    corrupt_skeleton,
    estimate_body_scale,
    make_jb_tensor,
    recompute_b_stream,
    recover_missing_linear,
    stable_seed,
)


JOINTS = list(NODE_INDEX)


def skeleton(frames=600, channels=2):
    """Create a finite, moving 18-joint skeleton with neck at the origin."""

    base2 = {
        "neck": (0.0, 0.0),
        "left_shoulder": (-0.20, 0.05),
        "right_shoulder": (0.20, 0.05),
        "left_hip": (-0.12, 0.55),
        "right_hip": (0.12, 0.55),
        "left_elbow": (-0.32, 0.25),
        "right_elbow": (0.32, 0.25),
        "left_wrist": (-0.40, 0.45),
        "right_wrist": (0.40, 0.45),
        "left_knee": (-0.14, 0.82),
        "right_knee": (0.14, 0.82),
        "left_ankle": (-0.16, 1.05),
        "right_ankle": (0.16, 1.05),
        "nose": (0.0, -0.18),
        "left_eye": (-0.04, -0.21),
        "right_eye": (0.04, -0.21),
        "left_ear": (-0.09, -0.18),
        "right_ear": (0.09, -0.18),
    }
    base = np.array([base2[name] for name in JOINTS], dtype=np.float64)
    if channels == 3:
        base = np.column_stack((base, np.linspace(-0.03, 0.03, len(base))))
        base[NODE_INDEX["neck"], 2] = 0.0
    X = np.repeat(base[None, :, :], frames, axis=0)
    phase = np.linspace(0, 4 * np.pi, frames)
    X[:, NODE_INDEX["left_wrist"], 0] += 0.03 * np.sin(phase)
    X[:, NODE_INDEX["right_ankle"], 1] += 0.02 * np.cos(phase)
    return X


class CorruptionTests(unittest.TestCase):
    def test_stable_seed(self):
        self.assertEqual(stable_seed(42, "x"), stable_seed(42, "x"))
        self.assertNotEqual(stable_seed(42, "x"), stable_seed(43, "x"))

    def test_body_scale_and_fallback(self):
        X = skeleton()
        scale, method, valid = estimate_body_scale(X, JOINTS)
        self.assertEqual(method, "neck_midhip")
        self.assertEqual(valid, len(X))
        self.assertGreater(scale, 0)

    def test_gaussian_is_deterministic_and_input_is_immutable(self):
        X = skeleton(frames=1200)
        original = X.copy()
        config = CorruptionConfig("gaussian", "moderate", seed=9)
        first = corrupt_skeleton(
            X, config, joint_names=JOINTS, subject_id="s", sample_id="clip"
        )
        second = corrupt_skeleton(
            X, config, joint_names=JOINTS, subject_id="s", sample_id="clip"
        )
        np.testing.assert_array_equal(X, original)
        np.testing.assert_array_equal(first.X_corrupt, second.X_corrupt)
        np.testing.assert_array_equal(first.corruption_mask, second.corruption_mask)
        self.assertTrue(first.corruption_mask.all())
        observed_std = (first.X_corrupt - X).std()
        expected_std = first.metadata["sigma"]
        self.assertAlmostEqual(observed_std, expected_std, delta=expected_std * 0.08)

    def test_paired_missing_masks_are_nested_and_exact(self):
        X = skeleton()
        mild = corrupt_skeleton(
            X,
            CorruptionConfig("missing", "mild", seed=10),
            joint_names=JOINTS,
            subject_id="s",
            sample_id="clip",
        )
        severe = corrupt_skeleton(
            X,
            CorruptionConfig("missing", "severe", seed=10),
            joint_names=JOINTS,
            subject_id="s",
            sample_id="clip",
        )
        self.assertTrue(np.all(~mild.corruption_mask | severe.corruption_mask))
        expected = round(0.03 * X.shape[0] * X.shape[1])
        self.assertEqual(int(mild.corruption_mask.sum()), expected)
        recovered, audit = recover_missing_linear(mild.X_corrupt)
        self.assertTrue(np.isfinite(recovered).all())
        self.assertGreater(audit["filled_values"], 0)

    def test_temporal_gap_is_contiguous(self):
        X = skeleton()
        result = corrupt_skeleton(
            X,
            CorruptionConfig(
                "missing", "moderate", seed=4, missing_submode="temporal_gap"
            ),
            joint_names=JOINTS,
            subject_id="s",
            sample_id="clip",
        )
        positions = np.argwhere(result.corruption_mask)
        self.assertGreater(len(positions), 0)
        self.assertEqual(len(np.unique(positions[:, 1])), 1)
        np.testing.assert_array_equal(
            positions[:, 0], np.arange(positions[0, 0], positions[-1, 0] + 1)
        )

    def test_paired_temporal_gaps_expand_around_same_center(self):
        X = skeleton()
        results = []
        for severity in ("mild", "moderate", "severe"):
            results.append(
                corrupt_skeleton(
                    X,
                    CorruptionConfig(
                        "missing",
                        severity,
                        seed=18,
                        missing_submode="temporal_gap",
                    ),
                    joint_names=JOINTS,
                    subject_id="s",
                    sample_id="clip",
                )
            )
        self.assertEqual(
            len({result.metadata["gap_joint"] for result in results}), 1
        )
        self.assertEqual(
            len({result.metadata["gap_center_frame"] for result in results}), 1
        )
        self.assertTrue(
            np.all(~results[0].corruption_mask | results[1].corruption_mask)
        )
        self.assertTrue(
            np.all(~results[1].corruption_mask | results[2].corruption_mask)
        )

    def test_scale_keeps_holmberg_neck_pivot(self):
        X = skeleton()
        result = corrupt_skeleton(
            X,
            CorruptionConfig("scale", "moderate", seed=2),
            joint_names=JOINTS,
            subject_id="s",
            sample_id="clip",
        )
        np.testing.assert_allclose(
            result.X_corrupt[:, NODE_INDEX["neck"]], 0.0, atol=1e-12
        )
        factor = result.metadata["scale_factor"]
        before = np.linalg.norm(
            X[:, NODE_INDEX["left_wrist"]] - X[:, NODE_INDEX["left_elbow"]], axis=1
        )
        after = np.linalg.norm(
            result.X_corrupt[:, NODE_INDEX["left_wrist"]]
            - result.X_corrupt[:, NODE_INDEX["left_elbow"]],
            axis=1,
        )
        np.testing.assert_allclose(after, before * factor)

    def test_rotation_preserves_distances_for_2d_and_3d(self):
        for channels in (2, 3):
            X = skeleton(channels=channels)
            result = corrupt_skeleton(
                X,
                CorruptionConfig("rotation", "severe", seed=13),
                joint_names=JOINTS,
                subject_id="s",
                sample_id=f"clip-{channels}",
            )
            before = np.linalg.norm(
                X[:, NODE_INDEX["left_hip"]] - X[:, NODE_INDEX["left_knee"]],
                axis=1,
            )
            after = np.linalg.norm(
                result.X_corrupt[:, NODE_INDEX["left_hip"]]
                - result.X_corrupt[:, NODE_INDEX["left_knee"]],
                axis=1,
            )
            np.testing.assert_allclose(after, before, atol=1e-12)

    def test_spike_only_changes_masked_sites(self):
        X = skeleton()
        result = corrupt_skeleton(
            X,
            CorruptionConfig("spike", "moderate", seed=22),
            joint_names=JOINTS,
            subject_id="s",
            sample_id="clip",
        )
        changed = np.any(result.X_corrupt != X, axis=2)
        np.testing.assert_array_equal(changed, result.corruption_mask)

    def test_b_stream_and_model_tensor(self):
        J = skeleton(frames=20)
        B = recompute_b_stream(J, JOINTS)
        np.testing.assert_array_equal(B[:, NODE_INDEX["neck"]], 0.0)
        np.testing.assert_allclose(
            B[:, NODE_INDEX["left_wrist"]],
            J[:, NODE_INDEX["left_wrist"]] - J[:, NODE_INDEX["left_elbow"]],
        )
        tensor = make_jb_tensor(J, B)
        self.assertEqual(tensor.shape, (4, 20, 18))
        self.assertTrue(np.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
