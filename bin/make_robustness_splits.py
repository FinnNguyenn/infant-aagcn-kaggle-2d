#!/usr/bin/env python
"""Freeze outer-CV plus inner-development subject manifests for robustness work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedShuffleSplit


def _age_bins(ages: pd.Series, max_bins: int, min_count: int = 2) -> np.ndarray:
    """Create stable approximate age strata, merging sparse quantile bins."""

    for bins in range(min(max_bins, len(ages) // min_count), 1, -1):
        try:
            labels = pd.qcut(ages, q=bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        if labels.nunique() >= 2 and labels.value_counts().min() >= min_count:
            return labels.to_numpy(dtype=int)
    return np.zeros(len(ages), dtype=int)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--age-file", required=True)
    parser.add_argument(
        "--output-dir", default="/kaggle/working/infant-aagcn-robustness-splits"
    )
    parser.add_argument("--k-folds", type=int, default=10)
    parser.add_argument("--fold", type=int, help="Generate one 1-indexed outer fold")
    parser.add_argument("--inner-dev-fraction", type=float, default=0.2)
    parser.add_argument("--age-bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.inner_dev_fraction < 1:
        parser.error("--inner-dev-fraction must be in (0,1)")
    if args.fold is not None and not 1 <= args.fold <= args.k_folds:
        parser.error("--fold must be between 1 and --k-folds")

    age_data = pd.read_csv(args.age_file, dtype={"test_id": str})
    required = {"test_id", "corrected_age", "outcome"}
    if not required.issubset(age_data.columns):
        raise ValueError(f"Age file must contain {sorted(required)}")
    age_data = age_data[age_data.outcome != 2].reset_index(drop=True)
    if age_data.test_id.duplicated().any():
        duplicates = age_data.loc[age_data.test_id.duplicated(False), "test_id"].tolist()
        raise ValueError(f"Duplicate infant IDs in age metadata: {duplicates[:5]}")

    rows: list[dict] = []
    outer = KFold(n_splits=args.k_folds, shuffle=False)
    generated_folds = []
    for fold, (outer_train_idx, outer_test_idx) in enumerate(outer.split(age_data), start=1):
        if args.fold is not None and fold != args.fold:
            continue
        generated_folds.append(fold)
        outer_train = age_data.iloc[outer_train_idx].reset_index(drop=True)
        strata = _age_bins(outer_train.corrected_age, args.age_bins)
        if np.unique(strata).size == 1:
            # Deterministic fallback when ages cannot form viable strata.
            rng = np.random.default_rng(args.seed + fold)
            order = rng.permutation(len(outer_train))
            dev_count = max(1, int(round(len(order) * args.inner_dev_fraction)))
            inner_dev_local = np.sort(order[:dev_count])
            inner_train_local = np.sort(order[dev_count:])
            split_strategy = "deterministic_random_fallback"
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=args.inner_dev_fraction,
                random_state=args.seed + fold,
            )
            inner_train_local, inner_dev_local = next(
                splitter.split(np.zeros(len(outer_train)), strata)
            )
            split_strategy = f"age_quantiles_{len(np.unique(strata))}"

        role_by_id = {
            **{str(value): "inner_train" for value in outer_train.iloc[inner_train_local].test_id},
            **{str(value): "inner_dev" for value in outer_train.iloc[inner_dev_local].test_id},
            **{str(value): "outer_test" for value in age_data.iloc[outer_test_idx].test_id},
        }
        if len(role_by_id) != len(age_data):
            raise RuntimeError(f"Fold {fold} has missing or overlapping subjects")
        for record in age_data.itertuples(index=False):
            rows.append(
                {
                    "fold": fold,
                    "subject_id": str(record.test_id),
                    "role": role_by_id[str(record.test_id)],
                    "age": float(record.corrected_age),
                    "outcome": int(record.outcome),
                    "split_seed": args.seed,
                    "split_strategy": split_strategy,
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["fold", "role", "subject_id"])
    csv_text = manifest.to_csv(index=False, lineterminator="\n")
    manifest_hash = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    output_dir = Path(args.output_dir).resolve()
    _atomic_text(output_dir / "split_manifest.csv", csv_text)
    config = {
        "age_file": str(Path(args.age_file).resolve()),
        "k_folds": args.k_folds,
        "generated_folds": generated_folds,
        "inner_dev_fraction": args.inner_dev_fraction,
        "requested_age_bins": args.age_bins,
        "seed": args.seed,
        "checkpoint_metric": "clean_inner_dev_MSE",
        "split_manifest_sha256": manifest_hash,
    }
    _atomic_text(
        output_dir / "split_config.json",
        json.dumps(config, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps({"output_dir": str(output_dir), **config}, indent=2))


if __name__ == "__main__":
    main()
