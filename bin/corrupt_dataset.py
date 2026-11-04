#!/usr/bin/env python
"""Create a fixed corrupted JB-AAGCN dataset/scenario bank.

The command never modifies the mounted clean dataset.  On Kaggle, point
``--output-root`` at ``/kaggle/working`` so every generated artifact remains
writable and can later be published as a Kaggle Dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from modules.corruption import (
    CorruptionConfig,
    CorruptionError,
    corrupt_skeleton,
    recompute_b_stream,
    recover_missing_linear,
)
from modules.robust_data import JOINT_NAMES, read_clean_j, write_jb_feather


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2).encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _load_existing_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def _write_manifest(path: Path, rows: dict[str, dict]) -> None:
    text = "\n".join(json.dumps(rows[key], sort_keys=True) for key in sorted(rows))
    if text:
        text += "\n"
    _atomic_bytes(path, text.encode("utf-8"))


def _filter_by_split(
    features: pd.DataFrame,
    manifest_path: str | None,
    fold: int | None,
    roles: set[str],
) -> pd.DataFrame:
    if manifest_path is None:
        return features
    if fold is None:
        raise ValueError("--fold is required with --split-manifest")
    split = pd.read_csv(manifest_path, dtype={"subject_id": str})
    required = {"fold", "subject_id", "role"}
    if not required.issubset(split.columns):
        raise ValueError(f"Split manifest must contain {sorted(required)}")
    selected = split[(split.fold == fold) & split.role.isin(roles)]
    ids = set(selected.subject_id.astype(str))
    if not ids:
        raise ValueError(f"No subjects for fold={fold}, roles={sorted(roles)}")
    return features[features.test_id.astype(str).isin(ids)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--age-file")
    parser.add_argument(
        "--output-root",
        default="/kaggle/working/infant-aagcn-corrupted",
        help="Writable root. Kaggle users should keep this under /kaggle/working.",
    )
    parser.add_argument("--scenario-name")
    parser.add_argument(
        "--family", required=True, choices=["gaussian", "spike", "missing", "scale", "rotation"]
    )
    parser.add_argument("--severity", required=True, choices=["mild", "moderate", "severe"])
    parser.add_argument(
        "--missing-submode",
        default="random_missing",
        choices=["random_missing", "temporal_gap"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--xy-data", action="store_true")
    parser.add_argument("--split-manifest")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--roles", default="inner_train,inner_dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-save-masks", action="store_true")
    args = parser.parse_args()

    scenario_name = args.scenario_name or (
        f"{args.family}-{args.severity}-r{args.replicate}-s{args.seed}"
    )
    scenario_root = Path(args.output_root).resolve() / scenario_name
    stream_kind = "streams_2d" if args.xy_data else "streams"
    output_streams = scenario_root / "data" / stream_kind / "combined"
    output_masks = scenario_root / "masks"
    output_streams.mkdir(parents=True, exist_ok=True)
    if not args.no_save_masks:
        output_masks.mkdir(parents=True, exist_ok=True)

    config = CorruptionConfig(
        family=args.family,
        severity=args.severity,
        seed=args.seed,
        replicate=args.replicate,
        missing_submode=args.missing_submode,
        fps=args.fps,
    )
    config_payload = {
        "schema_version": 1,
        "scenario_name": scenario_name,
        "data_dir": str(Path(args.data_dir).resolve()),
        "family": args.family,
        "severity": args.severity,
        "missing_submode": args.missing_submode,
        "seed": args.seed,
        "replicate": args.replicate,
        "fps": args.fps,
        "xy_data": args.xy_data,
        "fold": args.fold,
        "roles": sorted(set(args.roles.split(","))),
        "split_manifest": args.split_manifest,
        "recovery": "linear+nearest-boundary" if args.family == "missing" else None,
        "streams": ["j", "b"],
    }
    config_payload["config_sha256"] = hashlib.sha256(
        _json_bytes(config_payload)
    ).hexdigest()
    config_path = scenario_root / "scenario_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config_payload:
            raise RuntimeError(
                f"Output scenario already has a different config: {config_path}"
            )
    else:
        _atomic_bytes(config_path, _json_bytes(config_payload))

    features = pd.read_csv(args.feature_file, dtype={"test_id": str})
    required = {"segment", "test_id", "corrected_age"}
    if not required.issubset(features.columns):
        raise ValueError(f"Feature file must contain {sorted(required)}")
    roles = {role.strip() for role in args.roles.split(",") if role.strip()}
    features = _filter_by_split(features, args.split_manifest, args.fold, roles)
    if args.limit is not None:
        features = features.iloc[: args.limit].copy()
    if features.empty:
        raise RuntimeError("No segments selected")

    scenario_features = scenario_root / "data" / "features.csv"
    scenario_features.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(scenario_features, index=False)
    if args.age_file:
        destination = scenario_root / "metadata" / "combined.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.age_file, destination)

    manifest_path = scenario_root / "corruption_manifest.jsonl"
    manifest = _load_existing_manifest(manifest_path)
    input_dir = Path(args.data_dir)
    completed = 0
    for position, row in enumerate(features.itertuples(index=False), start=1):
        sample_id = str(row.segment)
        subject_id = str(row.test_id)
        source = input_dir / f"{sample_id}.feather"
        destination = output_streams / source.name
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists() and not args.overwrite and sample_id in manifest:
            completed += 1
            continue

        J, template = read_clean_j(source, args.xy_data)
        result = corrupt_skeleton(
            J,
            config,
            joint_names=JOINT_NAMES,
            subject_id=subject_id,
            sample_id=sample_id,
        )
        if result.status != "accepted":
            manifest[sample_id] = {
                "sample_id": sample_id,
                "subject_id": subject_id,
                "status": result.status,
                "reason": result.reason,
                "metadata": result.metadata,
            }
            _write_manifest(manifest_path, manifest)
            continue
        J_corrupt = result.X_corrupt
        recovery = None
        if args.family == "missing":
            J_corrupt, recovery = recover_missing_linear(J_corrupt)
        B_corrupt = recompute_b_stream(J_corrupt, JOINT_NAMES)
        if not np.isfinite(J_corrupt).all() or not np.isfinite(B_corrupt).all():
            raise CorruptionError(f"Non-finite model input for {sample_id}")
        write_jb_feather(destination, template, J_corrupt, B_corrupt)
        if not args.no_save_masks:
            np.savez_compressed(
                output_masks / f"{sample_id}.npz",
                observed_mask=result.observed_mask,
                corruption_mask=result.corruption_mask,
            )
        manifest[sample_id] = {
            "sample_id": sample_id,
            "subject_id": subject_id,
            "status": "accepted",
            "output_file": str(destination.relative_to(scenario_root)),
            "mask_file": None
            if args.no_save_masks
            else str((output_masks / f"{sample_id}.npz").relative_to(scenario_root)),
            "recovery": recovery,
            "metadata": result.metadata,
        }
        completed += 1
        if position == 1 or position % 50 == 0 or position == len(features):
            print(f"[{position}/{len(features)}] generated={completed} scenario={scenario_name}")
        if position % 25 == 0:
            _write_manifest(manifest_path, manifest)

    _write_manifest(manifest_path, manifest)
    summary = {
        "scenario_root": str(scenario_root),
        "selected_segments": int(len(features)),
        "accepted_or_existing": completed,
        "manifest_rows": len(manifest),
        "config_sha256": config_payload["config_sha256"],
    }
    _atomic_bytes(scenario_root / "summary.json", _json_bytes(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
