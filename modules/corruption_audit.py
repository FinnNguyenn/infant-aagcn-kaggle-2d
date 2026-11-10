"""Quantitative checks for motion-skeleton corruption.

The metrics in this module compare a corrupted/recovered skeleton against its
clean reference.  They are intentionally model-independent so they can be run
before committing GPU time to a robustness experiment.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def motion_energy(X: np.ndarray) -> float:
    """Mean squared first-difference energy per joint-frame."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[0] < 2:
        return float("nan")
    velocity = np.diff(X, axis=0)
    valid = np.isfinite(velocity).all(axis=2)
    if not valid.any():
        return float("nan")
    squared_speed = np.sum(velocity * velocity, axis=2)
    return float(np.mean(squared_speed[valid]))


def velocity_correlation(clean: np.ndarray, changed: np.ndarray) -> float:
    """Pearson correlation between flattened clean and changed velocities."""

    clean_velocity = np.diff(np.asarray(clean, dtype=float), axis=0).reshape(-1)
    changed_velocity = np.diff(np.asarray(changed, dtype=float), axis=0).reshape(-1)
    valid = np.isfinite(clean_velocity) & np.isfinite(changed_velocity)
    if valid.sum() < 2:
        return float("nan")
    clean_valid = clean_velocity[valid]
    changed_valid = changed_velocity[valid]
    if np.std(clean_valid) == 0 or np.std(changed_valid) == 0:
        return 1.0 if np.array_equal(clean_valid, changed_valid) else float("nan")
    return float(np.corrcoef(clean_valid, changed_valid)[0, 1])


def _safe_ratio(numerator: float, denominator: float, epsilon: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) <= epsilon:
        return float("nan")
    return float(numerator / denominator)


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def compare_skeletons(
    clean_j: np.ndarray,
    changed_j: np.ndarray,
    clean_b: np.ndarray,
    changed_b: np.ndarray,
    *,
    body_scale: float,
    corruption_mask: np.ndarray | None = None,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Return displacement, temporal, bone, and reconstruction diagnostics."""

    clean_j = np.asarray(clean_j, dtype=float)
    changed_j = np.asarray(changed_j, dtype=float)
    clean_b = np.asarray(clean_b, dtype=float)
    changed_b = np.asarray(changed_b, dtype=float)
    if clean_j.shape != changed_j.shape or clean_b.shape != changed_b.shape:
        raise ValueError("Clean and changed tensors must have matching shapes")
    if clean_j.shape != clean_b.shape:
        raise ValueError("J and B tensors must have matching shapes")
    if not np.isfinite(body_scale) or body_scale <= epsilon:
        raise ValueError("body_scale must be positive and finite")

    displacement = np.linalg.norm(changed_j - clean_j, axis=2) / body_scale
    clean_j_energy = motion_energy(clean_j)
    changed_j_energy = motion_energy(changed_j)
    clean_b_energy = motion_energy(clean_b)
    changed_b_energy = motion_energy(changed_b)

    clean_lengths = np.linalg.norm(clean_b, axis=2)
    changed_lengths = np.linalg.norm(changed_b, axis=2)
    valid_bones = np.isfinite(clean_lengths) & np.isfinite(changed_lengths)
    valid_bones &= clean_lengths > epsilon
    relative_bone_error = np.full(clean_lengths.shape, np.nan, dtype=float)
    relative_bone_error[valid_bones] = (
        np.abs(changed_lengths[valid_bones] - clean_lengths[valid_bones])
        / clean_lengths[valid_bones]
    )

    metrics: dict[str, Any] = {
        "displacement_mean_body_scale": float(np.nanmean(displacement)),
        "displacement_p50_body_scale": _percentile(displacement, 50),
        "displacement_p95_body_scale": _percentile(displacement, 95),
        "displacement_max_body_scale": _percentile(displacement, 100),
        "j_motion_energy_clean": clean_j_energy,
        "j_motion_energy_changed": changed_j_energy,
        "j_motion_energy_ratio": _safe_ratio(
            changed_j_energy, clean_j_energy, epsilon
        ),
        "j_velocity_correlation": velocity_correlation(clean_j, changed_j),
        "b_motion_energy_clean": clean_b_energy,
        "b_motion_energy_changed": changed_b_energy,
        "b_motion_energy_ratio": _safe_ratio(
            changed_b_energy, clean_b_energy, epsilon
        ),
        "b_velocity_correlation": velocity_correlation(clean_b, changed_b),
        "bone_length_relative_error_mean": float(
            np.nanmean(relative_bone_error)
        ),
        "bone_length_relative_error_p95": _percentile(
            relative_bone_error, 95
        ),
    }

    if corruption_mask is None:
        metrics.update(
            reconstruction_masked_count=0,
            reconstruction_rmse_body_scale=float("nan"),
            reconstruction_displacement_p95_body_scale=float("nan"),
        )
    else:
        mask = np.asarray(corruption_mask, dtype=bool)
        if mask.shape != clean_j.shape[:2]:
            raise ValueError("corruption_mask must have shape (T, J)")
        coordinate_mask = np.repeat(mask[..., None], clean_j.shape[2], axis=2)
        coordinate_error = (changed_j - clean_j) / body_scale
        masked_coordinate_error = coordinate_error[coordinate_mask]
        masked_displacement = displacement[mask]
        rmse = (
            float(np.sqrt(np.mean(masked_coordinate_error**2)))
            if masked_coordinate_error.size
            else float("nan")
        )
        metrics.update(
            reconstruction_masked_count=int(mask.sum()),
            reconstruction_rmse_body_scale=rmse,
            reconstruction_displacement_p95_body_scale=_percentile(
                masked_displacement, 95
            ),
        )

    return metrics
