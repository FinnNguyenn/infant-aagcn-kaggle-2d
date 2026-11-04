import tempfile
import unittest
from pathlib import Path

try:
    import feather
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from modules.constants import FEATURE_LIST, NODE_INDEX
    from modules.corruption import recompute_b_stream
    from modules.robust_data import (
        JOINT_NAMES,
        RobustInfantMotionDataset,
        coordinate_columns,
        write_jb_feather,
    )

    STACK_AVAILABLE = True
except ImportError:
    STACK_AVAILABLE = False


@unittest.skipUnless(STACK_AVAILABLE, "optional pandas/feather/torch stack unavailable")
class RobustDataTests(unittest.TestCase):
    def _skeleton(self, offset):
        channels = 2
        frame = np.zeros((18, channels), dtype=np.float64)
        for name, index in NODE_INDEX.items():
            frame[index] = (index * 0.02 + offset, index * 0.03)
        frame[NODE_INDEX["neck"]] = (0.0, 0.0)
        frame[NODE_INDEX["left_hip"]] = (-0.1, 0.5)
        frame[NODE_INDEX["right_hip"]] = (0.1, 0.5)
        frame[NODE_INDEX["left_shoulder"]] = (-0.2, 0.1)
        frame[NODE_INDEX["right_shoulder"]] = (0.2, 0.1)
        return np.repeat(frame[None, :, :], 600, axis=0)

    def _fixture(self, root):
        rows = []
        for number, age in (("s1", 80.0), ("s2", 110.0)):
            J = self._skeleton(0.01 if number == "s1" else 0.02)
            B = recompute_b_stream(J, JOINT_NAMES)
            template = pd.DataFrame(
                J.reshape(600, -1), columns=coordinate_columns(True)
            )
            write_jb_feather(root / f"{number}.feather", template, J, B)
            row = {
                "segment": number,
                "test_id": number,
                "corrected_age": age,
            }
            row.update({feature: 0.0 for feature in FEATURE_LIST})
            rows.append(row)
        return pd.DataFrame(rows)

    def test_labels_determinism_and_worker_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self._fixture(root)
            dataset = RobustInfantMotionDataset(
                root,
                data,
                xy_data=True,
                family_set="all",
                corruption_probability=1.0,
                seed=42,
                fold=1,
            )
            dataset.set_epoch(0)
            first, y_first, _ = dataset[0]
            again, _, _ = dataset[0]
            self.assertTrue(torch.equal(first, again))
            self.assertEqual(float(y_first), 80.0)
            self.assertEqual(float(dataset[1][1]), 110.0)

            loader0 = DataLoader(dataset, batch_size=1, num_workers=0)
            output0 = [batch[0] for batch in loader0]
            loader2 = DataLoader(dataset, batch_size=1, num_workers=2)
            output2 = [batch[0] for batch in loader2]
            self.assertEqual(len(output0), len(output2))
            for left, right in zip(output0, output2):
                self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
