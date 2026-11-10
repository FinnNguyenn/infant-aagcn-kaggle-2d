#!/usr/bin/env python
"""Audit corruption severity on real training segments before model training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modules.corruption import (
    CorruptionConfig,
    GAP_SECONDS,
    GAUSSIAN_ALPHA,
    MISSING_RATE,
    ROTATION_DEGREES,
    SCALE_BETA,
    SPIKE_MAGNITUDE,
    SPIKE_RATE,
    corrupt_skeleton,
    recover_missing_linear,
    recompute_b_stream,
)
from modules.corruption_audit import compare_skeletons
from modules.robust_data import JOINT_NAMES, read_clean_j


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(per_segment: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in per_segment.columns
        if column
        not in {
            "fold",
            "subject_id",
            "segment",
            "family",
            "severity",
            "replicate",
            "status",
            "reason",
            "scale_method",
        }
        and pd.api.types.is_numeric_dtype(per_segment[column])
    ]
    rows = []
    accepted = per_segment[per_segment.status == "accepted"]
    for (family, severity), frame in accepted.groupby(["family", "severity"]):
        row = {
            "family": family,
            "severity": severity,
            "accepted_segments": int(len(frame)),
            "rejected_segments": int(
                len(
                    per_segment[
                        (per_segment.family == family)
                        & (per_segment.severity == severity)
                        & (per_segment.status != "accepted")
                    ]
                )
            ),
        }
        for column in metric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            row[f"{column}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{column}_p95"] = (
                float(values.quantile(0.95)) if len(values) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--role", choices=["inner_train", "inner_dev"], default="inner_train")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument(
        "--sample-manifest",
        help="Optional prior sampled_segments.csv; reuses exactly those segments.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--corruption-seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument(
        "--families",
        default="gaussian,spike,missing,scale,rotation",
    )
    parser.add_argument("--severities", default="mild,moderate")
    parser.add_argument(
        "--missing-submode",
        choices=["random_missing", "temporal_gap"],
        default="random_missing",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--xy-data", action="store_true")
    args = parser.parse_args()

    if args.sample_size < 1 or args.replicates < 1:
        parser.error("--sample-size and --replicates must be positive")

    families = [value.strip() for value in args.families.split(",") if value.strip()]
    severities = [value.strip() for value in args.severities.split(",") if value.strip()]
    valid_families = {"gaussian", "spike", "missing", "scale", "rotation"}
    valid_severities = {"mild", "moderate", "severe"}
    if not families or not set(families).issubset(valid_families):
        parser.error(f"--families must be chosen from {sorted(valid_families)}")
    if not severities or not set(severities).issubset(valid_severities):
        parser.error(f"--severities must be chosen from {sorted(valid_severities)}")

    split = pd.read_csv(args.split_manifest, dtype={"subject_id": str})
    required_split = {"fold", "subject_id", "role"}
    if not required_split.issubset(split.columns):
        raise ValueError(f"Split manifest must contain {sorted(required_split)}")
    role_rows = split[(split.fold == args.fold) & (split.role == args.role)]
    subject_ids = set(role_rows.subject_id.astype(str))
    if not subject_ids:
        raise RuntimeError(f"No subjects for fold={args.fold}, role={args.role}")

    features = pd.read_csv(args.feature_file, dtype={"test_id": str})
    required_features = {"test_id", "segment"}
    if not required_features.issubset(features.columns):
        raise ValueError(f"Feature file must contain {sorted(required_features)}")
    candidates = features[features.test_id.astype(str).isin(subject_ids)].copy()
    candidates = candidates.drop_duplicates("segment").reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError("No feature rows match the selected split role")
    if args.sample_manifest:
        requested = pd.read_csv(args.sample_manifest, dtype={"test_id": str, "segment": str})
        required_sample = {"test_id", "segment"}
        if not required_sample.issubset(requested.columns):
            raise ValueError(f"Sample manifest must contain {sorted(required_sample)}")
        requested = requested[["test_id", "segment"]].drop_duplicates()
        sampled = requested.merge(
            candidates, on=["test_id", "segment"], how="left", indicator=True
        )
        if len(sampled) != len(requested) or (sampled._merge != "both").any():
            raise RuntimeError("Sample manifest contains segments outside the selected split role")
        sampled = sampled.drop(columns="_merge")
    else:
        sample_count = min(args.sample_size, len(candidates))
        sampled = candidates.sample(n=sample_count, random_state=args.sample_seed)
    sampled = sampled.sort_values(["test_id", "segment"]).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled[["test_id", "segment"]].to_csv(
        output_dir / "sampled_segments.csv", index=False
    )

    rows: list[dict] = []
    data_dir = Path(args.data_dir)
    total = len(sampled) * len(families) * len(severities) * args.replicates
    position = 0
    for record in sampled.itertuples(index=False):
        clean_j, _ = read_clean_j(data_dir / f"{record.segment}.feather", args.xy_data)
        clean_b = recompute_b_stream(clean_j, JOINT_NAMES)
        for family in families:
            for severity in severities:
                for replicate in range(args.replicates):
                    position += 1
                    config = CorruptionConfig(
                        family=family,
                        severity=severity,
                        seed=args.corruption_seed,
                        replicate=replicate,
                        missing_submode=args.missing_submode,
                        fps=args.fps,
                    )
                    result = corrupt_skeleton(
                        clean_j,
                        config,
                        joint_names=JOINT_NAMES,
                        subject_id=str(record.test_id),
                        sample_id=str(record.segment),
                    )
                    base = {
                        "fold": args.fold,
                        "subject_id": str(record.test_id),
                        "segment": str(record.segment),
                        "family": family,
                        "severity": severity,
                        "replicate": replicate,
                        "status": result.status,
                        "reason": result.reason,
                        "body_scale": result.metadata["body_scale"],
                        "scale_method": result.metadata["scale_method"],
                        "corrupted_joint_frames": int(result.corruption_mask.sum()),
                        "corrupted_joint_frame_fraction": float(result.corruption_mask.mean()),
                    }
                    if result.status == "accepted":
                        changed_j = result.X_corrupt
                        if family == "missing":
                            changed_j, recovery = recover_missing_linear(changed_j)
                            base.update(recovery)
                        changed_b = recompute_b_stream(changed_j, JOINT_NAMES)
                        base.update(
                            compare_skeletons(
                                clean_j,
                                changed_j,
                                clean_b,
                                changed_b,
                                body_scale=result.metadata["body_scale"],
                                corruption_mask=result.corruption_mask
                                if family == "missing"
                                else None,
                            )
                        )
                    rows.append(base)
                    if position == 1 or position % 100 == 0 or position == total:
                        print(f"[{position}/{total}] audited segment={record.segment}")

    per_segment = pd.DataFrame(rows)
    per_segment.to_csv(output_dir / "per_segment_metrics.csv", index=False)
    summary = _summary(per_segment)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    manifest_bytes = Path(args.split_manifest).read_bytes()
    config = {
        "args": vars(args),
        "selected_segments": int(len(sampled)),
        "candidate_segments": int(len(candidates)),
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "metric_notes": {
            "motion_energy": "mean squared first difference per joint-frame",
            "velocity_correlation": "Pearson correlation of flattened first differences",
            "displacement": "Euclidean joint displacement divided by clean body scale",
            "bone_length_distortion": "absolute relative error of non-zero physical bone lengths",
            "missing_reconstruction": "error after linear recovery at synthetically hidden sites",
        },
        "corruption_parameters": {
            "gaussian_alpha": GAUSSIAN_ALPHA,
            "spike_rate": SPIKE_RATE,
            "spike_magnitude": SPIKE_MAGNITUDE,
            "missing_rate": MISSING_RATE,
            "gap_seconds": GAP_SECONDS,
            "scale_beta": SCALE_BETA,
            "rotation_degrees": ROTATION_DEGREES,
        },
    }
    _write_json(output_dir / "audit_config.json", config)
    print(summary.to_string(index=False))
    print(f"Audit outputs written to {output_dir}")


if __name__ == "__main__":
    main()
