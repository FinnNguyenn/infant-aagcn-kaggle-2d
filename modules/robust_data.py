"""Feather adapters and an opt-in robust-training dataset.

The legacy :class:`modules.data.InfantMotionDataset` remains untouched.  This
module is only selected when robustness augmentation is explicitly enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import feather
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from modules.constants import FEATURE_LIST, NODE_INDEX
from modules.corruption import (
    CorruptionConfig,
    CorruptionError,
    corrupt_skeleton,
    make_jb_tensor,
    recompute_b_stream,
    recover_missing_linear,
    stable_seed,
)


JOINT_NAMES = list(NODE_INDEX)
FAMILY_SETS = {
    "local": ("gaussian", "spike", "missing"),
    "all": ("gaussian", "spike", "missing", "scale", "rotation"),
}
FAMILY_WEIGHTS = {
    "gaussian": 0.30,
    "spike": 0.20,
    "missing": 0.30,
    "scale": 0.10,
    "rotation": 0.10,
}


def _stream_frame(df_multi: pd.DataFrame, stream: str) -> pd.DataFrame:
    """Select the original augmentation and one stream from a repo Feather."""

    try:
        after_augmentation = df_multi.xs("original", level=0)
    except (KeyError, ValueError) as exc:
        raise CorruptionError("Feather does not contain the 'original' augmentation") from exc
    try:
        return after_augmentation.xs(stream, level=0).copy()
    except (KeyError, ValueError) as exc:
        raise CorruptionError(f"Feather does not contain stream {stream!r}") from exc


def coordinate_columns(xy_data: bool) -> list[str]:
    coords = ("x", "y") if xy_data else ("x", "y", "z")
    return [f"{joint}_{coord}" for joint in JOINT_NAMES for coord in coords]


def extract_original_j(
    df_multi: pd.DataFrame, xy_data: bool
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return ``J`` as ``(T,N,C)`` plus its index/column template."""

    frame = _stream_frame(df_multi, "j")
    columns = coordinate_columns(xy_data)
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise CorruptionError(f"J stream is missing columns: {absent[:5]}")
    frame = frame.loc[:, columns]
    C = 2 if xy_data else 3
    values = frame.to_numpy().reshape(len(frame), len(JOINT_NAMES), C)
    return values, frame


def build_jb_dataframe(
    template: pd.DataFrame, J: np.ndarray, B: np.ndarray
) -> pd.DataFrame:
    """Build a Feather-compatible MultiIndex frame containing only J and B."""

    if J.shape != B.shape or J.shape[0] != len(template):
        raise CorruptionError("J/B shapes do not match the source Feather template")
    j_frame = template.copy()
    b_frame = template.copy()
    j_frame.iloc[:, :] = J.reshape(len(template), -1)
    b_frame.iloc[:, :] = B.reshape(len(template), -1)
    return pd.concat({("original", "j"): j_frame, ("original", "b"): b_frame})


def read_clean_j(path: str | Path, xy_data: bool) -> tuple[np.ndarray, pd.DataFrame]:
    df_multi = feather.read_dataframe(str(path))
    return extract_original_j(df_multi, xy_data)


def write_jb_feather(
    path: str | Path, template: pd.DataFrame, J: np.ndarray, B: np.ndarray
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    feather.write_dataframe(build_jb_dataframe(template, J, B), str(tmp_path))
    tmp_path.replace(path)


class RobustInfantMotionDataset(Dataset):
    """One-file/one-label dataset with deterministic train-only corruption."""

    def __init__(
        self,
        directory: str | Path,
        data: pd.DataFrame,
        *,
        xy_data: bool,
        family_set: str,
        corruption_probability: float = 0.5,
        mild_probability: float = 0.7,
        seed: int = 42,
        fold: int = 0,
        fps: float = 30.0,
    ) -> None:
        if family_set not in FAMILY_SETS:
            raise ValueError(f"family_set must be one of {sorted(FAMILY_SETS)}")
        if not 0 <= corruption_probability <= 1:
            raise ValueError("corruption_probability must be in [0,1]")
        if not 0 <= mild_probability <= 1:
            raise ValueError("mild_probability must be in [0,1]")
        self.directory = Path(directory)
        self.data = data.reset_index(drop=True).copy()
        self.xy_data = xy_data
        self.family_set = family_set
        self.corruption_probability = corruption_probability
        self.mild_probability = mild_probability
        self.seed = int(seed)
        self.fold = int(fold)
        self.fps = float(fps)
        # Torch shared memory remains visible to persistent DataLoader workers
        # without deriving randomness from worker IDs or scheduling.
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch[0].item())

    def __len__(self) -> int:
        return len(self.data)

    def _choose_config(self, row: pd.Series) -> CorruptionConfig | None:
        subject_id = str(row.test_id)
        sample_id = str(row.segment)
        epoch = self.epoch
        rng = np.random.default_rng(
            stable_seed(
                self.seed,
                self.fold,
                epoch,
                subject_id,
                sample_id,
                "training-policy",
            )
        )
        if rng.random() >= self.corruption_probability:
            return None
        families = FAMILY_SETS[self.family_set]
        weights = np.array([FAMILY_WEIGHTS[name] for name in families], dtype=float)
        weights /= weights.sum()
        family = str(rng.choice(families, p=weights))
        severity = "mild" if rng.random() < self.mild_probability else "moderate"
        epoch_seed = stable_seed(self.seed, self.fold, epoch, "training-corruption")
        return CorruptionConfig(
            family=family,
            severity=severity,
            seed=epoch_seed,
            fps=self.fps,
            min_valid_scale_frames=30,
        )

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        path = self.directory / f"{row.segment}.feather"
        J, _ = read_clean_j(path, self.xy_data)
        config = self._choose_config(row)
        if config is not None:
            result = corrupt_skeleton(
                J,
                config,
                joint_names=JOINT_NAMES,
                subject_id=str(row.test_id),
                sample_id=str(row.segment),
            )
            if result.status != "accepted":
                raise CorruptionError(
                    f"Corruption rejected for {row.segment}: {result.reason}"
                )
            J = result.X_corrupt
            if config.family == "missing":
                J, _ = recover_missing_linear(J)
        B = recompute_b_stream(J, JOINT_NAMES)
        tensor = make_jb_tensor(J, B)
        if not np.isfinite(tensor).all():
            raise CorruptionError(f"Non-finite JB tensor for {row.segment}")

        X = torch.tensor(tensor, dtype=torch.float32)
        y = torch.tensor(float(row.corrected_age), dtype=torch.float32)
        fts = torch.tensor(row[FEATURE_LIST].to_numpy(dtype=float), dtype=torch.float32)
        return X, y, fts
