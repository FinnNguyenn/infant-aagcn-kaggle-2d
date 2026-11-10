#!/usr/bin/env python
"""Evaluate one frozen checkpoint on clean and fixed corrupted inner-dev data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from modules.constants import (
    COORD_EDGE_INDEX,
    FC_EDGE_INDEX,
    FEATURE_LIST,
    N,
    PHYS_EDGE_INDEX,
)
from modules.corruption import (
    CorruptionConfig,
    corrupt_skeleton,
    make_jb_tensor,
    recover_missing_linear,
    recompute_b_stream,
)
from modules.data import InfantMotionDataset
from modules.model import AdaptiveSTGCN
from modules.robust_data import JOINT_NAMES, read_clean_j


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FixedCorruptionDataset(Dataset):
    """Apply one deterministic scenario without changing row membership."""

    def __init__(
        self,
        data_dir: str | Path,
        data: pd.DataFrame,
        *,
        xy_data: bool,
        config: CorruptionConfig,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data = data.reset_index(drop=True).copy()
        self.xy_data = xy_data
        self.config = config

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        clean_j, _ = read_clean_j(
            self.data_dir / f"{row.segment}.feather", self.xy_data
        )
        result = corrupt_skeleton(
            clean_j,
            self.config,
            joint_names=JOINT_NAMES,
            subject_id=str(row.test_id),
            sample_id=str(row.segment),
        )
        if result.status != "accepted":
            raise RuntimeError(
                f"Scenario rejected segment {row.segment}: {result.reason}"
            )
        changed_j = result.X_corrupt
        if self.config.family == "missing":
            changed_j, _ = recover_missing_linear(changed_j)
        changed_b = recompute_b_stream(changed_j, JOINT_NAMES)
        X = torch.tensor(make_jb_tensor(changed_j, changed_b), dtype=torch.float32)
        y = torch.tensor(float(row.corrected_age), dtype=torch.float32)
        fts = torch.tensor(
            row[FEATURE_LIST].to_numpy(dtype=float), dtype=torch.float32
        )
        return X, y, fts


def _edge_index(name: str) -> torch.Tensor:
    choices = {
        "phys": PHYS_EDGE_INDEX,
        "coord": COORD_EDGE_INDEX,
        "fc": FC_EDGE_INDEX,
    }
    if name not in choices:
        raise ValueError(f"Unknown edge type: {name}")
    return torch.tensor(choices[name])


def _predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for X, y, fts in loader:
            output = model(X.to(device), fts.to(device)).squeeze(-1)
            predictions.append(output.detach().cpu().reshape(-1))
            targets.append(y.detach().cpu().reshape(-1))
    return torch.cat(targets).numpy(), torch.cat(predictions).numpy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error) / np.maximum(np.abs(y_true), 1e-8)) * 100),
        "bias": float(np.mean(error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Experiment root containing fold_N")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--families", default="gaussian,spike,missing")
    parser.add_argument("--severities", default="mild")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--corruption-seed", type=int, default=42)
    parser.add_argument(
        "--missing-submode",
        choices=["random_missing", "temporal_gap"],
        default="random_missing",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    if args.replicates < 1:
        parser.error("--replicates must be positive")
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    severities = [item.strip() for item in args.severities.split(",") if item.strip()]

    fold_dir = Path(args.model_dir) / f"fold_{args.fold}"
    metadata_path = fold_dir / "metadata.json"
    checkpoint_path = fold_dir / "best_model.ckpt"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Expected {metadata_path} and {checkpoint_path}; finish training this fold first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_args = metadata["args"]
    streams = train_args["streams"].split(",")
    if streams != ["j", "b"]:
        raise ValueError("Fixed corruption evaluation currently requires --streams j,b")
    xy_data = bool(train_args["xy_data"])

    split = pd.read_csv(args.split_manifest, dtype={"subject_id": str})
    dev_ids = set(
        split[
            (split.fold == args.fold) & (split.role == "inner_dev")
        ].subject_id.astype(str)
    )
    if not dev_ids:
        raise RuntimeError(f"No inner_dev subjects for fold {args.fold}")
    features = pd.read_csv(args.feature_file, dtype={"test_id": str})
    dev_features = features[features.test_id.astype(str).isin(dev_ids)].copy()
    dev_features = dev_features.reset_index(drop=True)
    if dev_features.empty:
        raise RuntimeError("No feature rows match inner_dev")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = AdaptiveSTGCN.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        edge_index=_edge_index(train_args["edges"]),
        num_nodes=N,
        in_channels=(2 if xy_data else 3) * len(streams),
        learning_rate=train_args["learning_rate"],
        adaptive=train_args["adaptive"],
        attention=train_args["attention"],
        masking=train_args["masking"],
        concat_features=train_args["concat_features"],
        hidden_dim=train_args["hidden_dim"],
        kt=train_args["kt"],
        map_location=device,
    ).to(device)

    scenario_specs: list[tuple[str, CorruptionConfig | None]] = [("clean", None)]
    for family in families:
        for severity in severities:
            for replicate in range(args.replicates):
                name = f"{family}-{severity}-r{replicate}"
                scenario_specs.append(
                    (
                        name,
                        CorruptionConfig(
                            family=family,
                            severity=severity,
                            seed=args.corruption_seed,
                            replicate=replicate,
                            missing_submode=args.missing_submode,
                            fps=args.fps,
                        ),
                    )
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    prediction_frames = []
    clean_predictions = None
    for scenario_name, config in scenario_specs:
        if config is None:
            dataset = InfantMotionDataset(
                args.data_dir,
                dev_features,
                streams,
                xy_data=xy_data,
                predict=True,
            )
        else:
            dataset = FixedCorruptionDataset(
                args.data_dir,
                dev_features,
                xy_data=xy_data,
                config=config,
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        y_true, y_pred = _predict(model, loader, device)
        if clean_predictions is None:
            clean_predictions = y_pred.copy()
        row = {
            "fold": args.fold,
            "scenario": scenario_name,
            "family": "clean" if config is None else config.family,
            "severity": "clean" if config is None else config.severity,
            "replicate": -1 if config is None else config.replicate,
            "segments": int(len(y_true)),
            **_metrics(y_true, y_pred),
            "prediction_drift_mae_vs_clean": float(
                np.mean(np.abs(y_pred - clean_predictions))
            ),
        }
        metric_rows.append(row)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": args.fold,
                    "scenario": scenario_name,
                    "segment": dev_features.segment.astype(str),
                    "test_id": dev_features.test_id.astype(str),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "absolute_error": np.abs(y_pred - y_true),
                    "prediction_drift_vs_clean": y_pred - clean_predictions,
                }
            )
        )
        print(
            f"{scenario_name}: MAE={row['mae']:.4f}, "
            f"RMSE={row['rmse']:.4f}, drift={row['prediction_drift_mae_vs_clean']:.4f}"
        )

    metrics_frame = pd.DataFrame(metric_rows)
    clean_mae = float(metrics_frame.loc[metrics_frame.scenario == "clean", "mae"].iloc[0])
    metrics_frame["mae_delta_vs_clean"] = metrics_frame.mae - clean_mae
    metrics_frame["mae_relative_change_vs_clean"] = (
        metrics_frame["mae_delta_vs_clean"] / max(clean_mae, 1e-8)
    )
    metrics_frame.to_csv(output_dir / "scenario_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False
    )

    config_output = {
        "args": vars(args),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "split_manifest_sha256": _sha256_file(Path(args.split_manifest)),
        "evaluated_role": "inner_dev",
        "outer_test_accessed": False,
    }
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config_output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluation outputs written to {output_dir}")


if __name__ == "__main__":
    main()
