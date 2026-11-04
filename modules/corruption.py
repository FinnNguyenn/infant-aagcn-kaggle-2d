"""Deterministic skeleton corruptions for the infant robustness benchmark.

The functions in this module operate on a clean joint stream with shape
``(time, joints, coordinates)``.  They deliberately do not depend on torch or
pandas so the mathematical core can be tested independently of the training
and Feather I/O stacks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from modules.constants import NODE_INDEX, PHYS_EDGE_LABELS


FAMILIES = ("gaussian", "spike", "missing", "scale", "rotation")
SEVERITIES = ("mild", "moderate", "severe")

GAUSSIAN_ALPHA = {"mild": 0.005, "moderate": 0.015, "severe": 0.03}
SPIKE_RATE = {"mild": 0.005, "moderate": 0.02, "severe": 0.05}
SPIKE_MAGNITUDE = {
    "mild": (0.05, 0.15),
    "moderate": (0.10, 0.30),
    "severe": (0.20, 0.50),
}
MISSING_RATE = {"mild": 0.03, "moderate": 0.08, "severe": 0.15}
GAP_SECONDS = {
    "mild": (0.08, 0.12),
    "moderate": (0.16, 0.28),
    "severe": (0.32, 0.48),
}
SCALE_BETA = {"mild": 0.05, "moderate": 0.10, "severe": 0.20}
ROTATION_DEGREES = {"mild": 5.0, "moderate": 10.0, "severe": 20.0}


class CorruptionError(ValueError):
    """Raised when a sample cannot be safely corrupted or recovered."""


@dataclass(frozen=True)
class CorruptionConfig:
    family: str
    severity: str
    seed: int = 42
    replicate: int = 0
    missing_submode: str = "random_missing"
    pivot_mode: str = "median_neck"
    affected_joint_rate: float = 1.0
    fps: float = 30.0
    strict_fov: bool = False
    min_valid_scale_frames: int = 30
    min_valid_scale_fraction: float = 0.05
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown corruption family: {self.family!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"Unknown severity: {self.severity!r}")
        if self.missing_submode not in {"random_missing", "temporal_gap"}:
            raise ValueError(
                "missing_submode must be 'random_missing' or 'temporal_gap'"
            )
        if not 0 < self.affected_joint_rate <= 1:
            raise ValueError("affected_joint_rate must be in (0, 1]")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")


@dataclass
class CorruptionResult:
    X_corrupt: np.ndarray
    observed_mask: np.ndarray
    corruption_mask: np.ndarray
    confidence_corrupt: np.ndarray | None
    metadata: dict[str, Any]
    status: str = "accepted"
    reason: str | None = None


def stable_seed(*parts: Any) -> int:
    """Return a process-independent uint64 seed for the supplied values."""

    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _rng(config: CorruptionConfig, subject_id: str, sample_id: str, stream: str):
    # Severity is intentionally absent so mild/moderate/severe share latent
    # randomness during paired-severity evaluation.
    seed = stable_seed(
        config.seed,
        subject_id,
        sample_id,
        config.family,
        config.replicate,
        stream,
    )
    return np.random.default_rng(seed), seed


def _validate_skeleton(X: np.ndarray, joint_names: Sequence[str]) -> np.ndarray:
    array = np.asarray(X)
    if array.ndim != 3 or array.shape[2] not in (2, 3):
        raise CorruptionError("Skeleton must have shape (T, J, 2|3)")
    if array.shape[0] < 2:
        raise CorruptionError("Skeleton must contain at least two frames")
    if array.shape[1] != len(joint_names):
        raise CorruptionError("joint_names length does not match skeleton")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    return array


def _joint_index(joint_names: Sequence[str]) -> dict[str, int]:
    index = {name: i for i, name in enumerate(joint_names)}
    if len(index) != len(joint_names):
        raise CorruptionError("joint_names contains duplicates")
    return index


def _valid_distances(
    X: np.ndarray, start: np.ndarray, end: np.ndarray, epsilon: float
) -> np.ndarray:
    distances = np.linalg.norm(start - end, axis=1)
    valid = np.isfinite(start).all(axis=1) & np.isfinite(end).all(axis=1)
    valid &= np.isfinite(distances) & (distances > epsilon)
    return distances[valid]


def estimate_body_scale(
    X: np.ndarray,
    joint_names: Sequence[str],
    *,
    min_valid_frames: int = 30,
    min_valid_fraction: float = 0.05,
    epsilon: float = 1e-6,
) -> tuple[float, str, int]:
    """Estimate a robust anatomical scale using the locked v4 fallback order."""

    X = _validate_skeleton(X, joint_names)
    idx = _joint_index(joint_names)
    required = max(min_valid_frames, int(math.ceil(X.shape[0] * min_valid_fraction)))

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    if {"neck", "left_hip", "right_hip"}.issubset(idx):
        midhip = (X[:, idx["left_hip"]] + X[:, idx["right_hip"]]) / 2.0
        candidates.append(("neck_midhip", X[:, idx["neck"]], midhip))
    if {"left_shoulder", "right_shoulder"}.issubset(idx):
        candidates.append(
            (
                "shoulder_width",
                X[:, idx["left_shoulder"]],
                X[:, idx["right_shoulder"]],
            )
        )
    if {"left_hip", "right_hip"}.issubset(idx):
        candidates.append(
            ("hip_width", X[:, idx["left_hip"]], X[:, idx["right_hip"]])
        )

    for method, start, end in candidates:
        distances = _valid_distances(X, start, end, epsilon)
        if distances.size >= required:
            scale = float(np.median(distances))
            if np.isfinite(scale) and scale > epsilon:
                return scale, method, int(distances.size)

    raise CorruptionError(
        f"invalid_body_scale: fewer than {required} valid frames for every method"
    )


def estimate_pivot(
    X: np.ndarray, joint_names: Sequence[str], mode: str = "median_neck"
) -> np.ndarray:
    """Estimate one fixed pivot for the complete segment."""

    X = _validate_skeleton(X, joint_names)
    idx = _joint_index(joint_names)
    if mode == "median_neck":
        if "neck" not in idx:
            raise CorruptionError("median_neck pivot requires the neck joint")
        points = X[:, idx["neck"]]
    elif mode == "median_midhip":
        if not {"left_hip", "right_hip"}.issubset(idx):
            raise CorruptionError("median_midhip pivot requires both hip joints")
        points = (X[:, idx["left_hip"]] + X[:, idx["right_hip"]]) / 2.0
    elif mode == "median_body_center":
        points = np.nanmean(X, axis=1)
    else:
        raise CorruptionError(f"Unsupported pivot mode: {mode!r}")
    pivot = np.nanmedian(points, axis=0)
    if not np.isfinite(pivot).all():
        raise CorruptionError(f"Unable to estimate finite {mode} pivot")
    return pivot


def _fov_valid(X: np.ndarray, frame_size: tuple[int, int] | None) -> bool:
    if frame_size is None:
        return False
    height, width = frame_size
    finite = np.isfinite(X).all(axis=2)
    points = X[finite]
    if not points.size:
        return False
    return bool(
        (points[:, 0] >= 0).all()
        and (points[:, 0] < width).all()
        and (points[:, 1] >= 0).all()
        and (points[:, 1] < height).all()
    )


def corrupt_skeleton(
    X: np.ndarray,
    config: CorruptionConfig,
    *,
    joint_names: Sequence[str],
    subject_id: str,
    sample_id: str,
    confidence: np.ndarray | None = None,
    frame_size: tuple[int, int] | None = None,
) -> CorruptionResult:
    """Apply one deterministic corruption family to a clean skeleton."""

    clean = _validate_skeleton(X, joint_names)
    corrupted = clean.copy()
    observed = np.isfinite(clean).all(axis=2)
    corruption_mask = np.zeros(observed.shape, dtype=bool)
    confidence_corrupt = None if confidence is None else np.asarray(confidence).copy()
    if confidence_corrupt is not None and confidence_corrupt.shape != observed.shape:
        raise CorruptionError("confidence must have shape (T, J)")

    scale, scale_method, valid_scale_frames = estimate_body_scale(
        clean,
        joint_names,
        min_valid_frames=config.min_valid_scale_frames,
        min_valid_fraction=config.min_valid_scale_fraction,
        epsilon=config.epsilon,
    )
    metadata: dict[str, Any] = {
        "family": config.family,
        "severity": config.severity,
        "subject_id": str(subject_id),
        "sample_id": str(sample_id),
        "replicate": config.replicate,
        "body_scale": scale,
        "scale_method": scale_method,
        "valid_scale_frames": valid_scale_frames,
        "post_segmentation_corruption": True,
        "config": asdict(config),
    }

    if config.family == "gaussian":
        noise_rng, latent_seed = _rng(config, subject_id, sample_id, "gaussian")
        mask_rng, _ = _rng(config, subject_id, sample_id, "gaussian-mask")
        noise = noise_rng.normal(size=clean.shape)
        affected = observed.copy()
        if config.affected_joint_rate < 1.0:
            affected &= mask_rng.random(observed.shape) < config.affected_joint_rate
        sigma = GAUSSIAN_ALPHA[config.severity] * scale
        corrupted[affected] += noise[affected] * sigma
        corruption_mask = affected
        metadata.update(alpha=GAUSSIAN_ALPHA[config.severity], sigma=sigma)

    elif config.family == "spike":
        site_rng, latent_seed = _rng(config, subject_id, sample_id, "spike-sites")
        direction_rng, _ = _rng(config, subject_id, sample_id, "spike-direction")
        magnitude_rng, _ = _rng(config, subject_id, sample_id, "spike-magnitude")
        rate = SPIKE_RATE[config.severity]
        corruption_mask = observed & (site_rng.random(observed.shape) < rate)
        directions = direction_rng.normal(size=clean.shape)
        norms = np.linalg.norm(directions, axis=2, keepdims=True)
        directions /= np.where(norms > 0, norms, 1.0)
        low, high = SPIKE_MAGNITUDE[config.severity]
        magnitudes = (low + (high - low) * magnitude_rng.random(observed.shape)) * scale
        delta = directions * magnitudes[..., None]
        corrupted[corruption_mask] += delta[corruption_mask]
        metadata.update(
            spike_rate=rate,
            spike_magnitude_range_body_scale=[low, high],
            spike_count=int(corruption_mask.sum()),
        )

    elif config.family == "missing":
        if config.missing_submode == "random_missing":
            site_rng, latent_seed = _rng(config, subject_id, sample_id, "missing-sites")
            observed_flat = np.flatnonzero(observed.ravel())
            count = int(round(MISSING_RATE[config.severity] * observed_flat.size))
            scores = site_rng.random(observed_flat.size)
            chosen = observed_flat[np.argsort(scores, kind="stable")[:count]]
            corruption_mask.ravel()[chosen] = True
            metadata.update(
                missing_submode="random_missing",
                requested_missing_rate=MISSING_RATE[config.severity],
            )
        else:
            gap_rng, latent_seed = _rng(config, subject_id, sample_id, "temporal-gap")
            idx = _joint_index(joint_names)
            preferred = [
                name
                for name in (
                    "left_wrist",
                    "right_wrist",
                    "left_ankle",
                    "right_ankle",
                    "left_elbow",
                    "right_elbow",
                    "left_knee",
                    "right_knee",
                )
                if name in idx
            ]
            if not preferred:
                preferred = list(joint_names)
            joint_name = preferred[int(gap_rng.integers(len(preferred)))]
            center_fraction = float(gap_rng.random())
            low_s, high_s = GAP_SECONDS[config.severity]
            duration_s = float(gap_rng.uniform(low_s, high_s))
            gap_frames = max(1, int(round(duration_s * config.fps)))
            gap_frames = min(gap_frames, clean.shape[0])
            # Keep the same latent center across paired severities; only the
            # duration expands.  Clipping at a boundary preserves nesting.
            center = int(round(center_fraction * (clean.shape[0] - 1)))
            start = max(0, min(clean.shape[0] - gap_frames, center - gap_frames // 2))
            end = start + gap_frames
            j = idx[joint_name]
            corruption_mask[start:end, j] = observed[start:end, j]
            metadata.update(
                missing_submode="temporal_gap",
                gap_joint=joint_name,
                gap_start=start,
                gap_end=end,
                gap_center_frame=center,
                gap_frames=gap_frames,
                gap_duration_seconds=gap_frames / config.fps,
            )
        corrupted[corruption_mask] = np.nan
        if confidence_corrupt is not None:
            confidence_corrupt[corruption_mask] = 0.0
        metadata.update(
            missing_count=int(corruption_mask.sum()),
            missing_fraction=float(corruption_mask.sum() / max(1, observed.sum())),
        )

    elif config.family == "scale":
        transform_rng, latent_seed = _rng(config, subject_id, sample_id, "scale")
        pivot = estimate_pivot(clean, joint_names, config.pivot_mode)
        u = float(transform_rng.uniform(-1.0, 1.0))
        factor = 1.0 + SCALE_BETA[config.severity] * u
        corrupted = pivot + factor * (clean - pivot)
        corruption_mask = observed.copy()
        metadata.update(
            pivot=pivot.tolist(), scale_factor=factor, scale_beta=SCALE_BETA[config.severity]
        )

    else:  # rotation
        transform_rng, latent_seed = _rng(config, subject_id, sample_id, "rotation")
        pivot = estimate_pivot(clean, joint_names, config.pivot_mode)
        u = float(transform_rng.uniform(-1.0, 1.0))
        angle_deg = ROTATION_DEGREES[config.severity] * u
        angle = np.deg2rad(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        if clean.shape[2] == 2:
            matrix = np.array([[c, -s], [s, c]], dtype=clean.dtype)
        else:
            matrix = np.array(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                dtype=clean.dtype,
            )
        corrupted = (clean - pivot) @ matrix.T + pivot
        corruption_mask = observed.copy()
        metadata.update(
            pivot=pivot.tolist(),
            rotation_axis="z",
            theta_degrees=angle_deg,
            rotation_matrix=matrix.tolist(),
        )

    metadata["latent_seed"] = latent_seed
    metadata["corrupted_joint_frames"] = int(corruption_mask.sum())

    if config.strict_fov and not _fov_valid(corrupted, frame_size):
        return CorruptionResult(
            corrupted,
            observed,
            corruption_mask,
            confidence_corrupt,
            metadata,
            status="skipped",
            reason="fov_violation",
        )

    return CorruptionResult(
        corrupted,
        observed,
        corruption_mask,
        confidence_corrupt,
        metadata,
    )


def recover_missing_linear(X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Linearly fill internal NaNs and nearest-fill boundary NaNs.

    ``numpy.interp`` has exactly these boundary semantics.  A fully missing
    joint trajectory is rejected because fabricating a body-part location
    would make the benchmark incomparable with the clean reference.
    """

    recovered = np.asarray(X).copy()
    if recovered.ndim != 3:
        raise CorruptionError("Skeleton must have shape (T, J, C)")
    x = np.arange(recovered.shape[0])
    filled = 0
    for joint in range(recovered.shape[1]):
        joint_valid = np.isfinite(recovered[:, joint]).all(axis=1)
        if not joint_valid.any():
            raise CorruptionError(f"all_missing_joint:{joint}")
        for channel in range(recovered.shape[2]):
            signal = recovered[:, joint, channel]
            valid = np.isfinite(signal)
            if not valid.any():
                raise CorruptionError(f"all_missing_joint_channel:{joint}:{channel}")
            missing = ~valid
            if missing.any():
                signal[missing] = np.interp(x[missing], x[valid], signal[valid])
                filled += int(missing.sum())
    if not np.isfinite(recovered).all():
        raise CorruptionError("Recovery produced a non-finite skeleton")
    return recovered, {"recovery_method": "linear", "filled_values": filled}


def recompute_b_stream(
    J: np.ndarray,
    joint_names: Sequence[str] | None = None,
    edges: Iterable[tuple[str, str]] = PHYS_EDGE_LABELS,
) -> np.ndarray:
    """Recompute bone vectors in the same joint order as the J stream."""

    names = list(NODE_INDEX) if joint_names is None else list(joint_names)
    J = _validate_skeleton(J, names)
    idx = _joint_index(names)
    B = np.zeros_like(J)
    assigned = {"neck"} if "neck" in idx else set()
    for source, target in edges:
        if source not in idx or target not in idx:
            raise CorruptionError(f"Edge {source!r}->{target!r} is absent from joint schema")
        B[:, idx[target]] = J[:, idx[target]] - J[:, idx[source]]
        assigned.add(target)
    missing = set(names) - assigned
    if missing:
        raise CorruptionError(f"No physical parent edge for joints: {sorted(missing)}")
    return B


def make_jb_tensor(J: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return model input with the legacy channel order ``C*streams,T,J``."""

    if J.shape != B.shape:
        raise CorruptionError("J and B must have identical shapes")
    T, N, C = J.shape
    data = np.stack((J, B), axis=-1)
    return data.transpose(2, 0, 1, 3).reshape(C * 2, T, N)
