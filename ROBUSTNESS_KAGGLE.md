# Controlled corruption robustness on Kaggle

This workflow implements the locked v4 pilot for the 2D `JB-AAGCN` model. It
keeps the mounted clean dataset read-only and writes split manifests,
corrupted Feather files, masks, audit metadata, and model results under
`/kaggle/working`.

The pilot supports `J+B` only. It does not synthesize or mix stale `V/A`
streams.

## 1. Setup

```bash
!git clone https://github.com/FinnNguyenn/infant-aagcn-kaggle-2d.git /kaggle/working/infant-aagcn
%cd /kaggle/working/infant-aagcn
%pip install -q -r requirements-kaggle.txt
```

Set the mounted clean dataset root. Adjust this path to match the Kaggle
Dataset slug:

```python
%env CLEAN_ROOT=/kaggle/input/infant-aagcn-2d-data
%env WORK_ROOT=/kaggle/working/infant-robustness

import os
from pathlib import Path

CLEAN_ROOT = Path(os.environ['CLEAN_ROOT'])
WORK_ROOT = Path(os.environ['WORK_ROOT'])

assert (CLEAN_ROOT / 'metadata/combined.csv').is_file()
assert (CLEAN_ROOT / 'data/features.csv').is_file()
assert list((CLEAN_ROOT / 'data/streams_2d/combined').glob('*.feather'))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
```

## 2. Freeze outer/inner subject splits

The original outer folds are preserved. Each outer-training fold gets one
subject-level, age-aware inner development holdout. The outer test subjects are
recorded in the manifest but are not loaded by `bin/train.py` during model
development.

```bash
!PYTHONPATH=/kaggle/working/infant-aagcn python bin/make_robustness_splits.py \
  --age-file "$CLEAN_ROOT/metadata/combined.csv" \
  --output-dir "$WORK_ROOT/splits" \
  --k-folds 10 \
  --inner-dev-fraction 0.20 \
  --age-bins 5 \
  --seed 42
```

The main output is:

```text
/kaggle/working/infant-robustness/splits/split_manifest.csv
```

## 3. Generate a fixed corrupted dataset in working

This example generates a moderate Gaussian scenario for fold 1, restricted to
`inner_train` and `inner_dev`. The clean input under `/kaggle/input` is never
modified.

```bash
!PYTHONPATH=/kaggle/working/infant-aagcn python bin/corrupt_dataset.py \
  --data-dir "$CLEAN_ROOT/data/streams_2d/combined" \
  --feature-file "$CLEAN_ROOT/data/features.csv" \
  --age-file "$CLEAN_ROOT/metadata/combined.csv" \
  --split-manifest "$WORK_ROOT/splits/split_manifest.csv" \
  --fold 1 \
  --roles inner_train,inner_dev \
  --family gaussian \
  --severity moderate \
  --seed 42 \
  --replicate 0 \
  --xy-data \
  --output-root "$WORK_ROOT/scenarios"
```

The resulting directory is directly reusable by the legacy `train.py` path:

```text
/kaggle/working/infant-robustness/scenarios/gaussian-moderate-r0-s42/
├── scenario_config.json
├── summary.json
├── corruption_manifest.jsonl
├── masks/*.npz
├── metadata/combined.csv
└── data/
    ├── features.csv
    └── streams_2d/combined/*.feather
```

For a quick smoke test, add `--limit 8`. Other supported scenarios use
`--family spike|missing|scale|rotation`. Missing also accepts
`--missing-submode random_missing|temporal_gap`.

## 4. Train P0 on clean data

The checkpoint is selected by clean `inner_dev` MSE, matching the legacy
`val_loss` rule. Only fold 1 is shown:

```bash
!PYTHONPATH=/kaggle/working/infant-aagcn python bin/train.py \
  --data-dir "$CLEAN_ROOT/data/streams_2d/combined" \
  --age-file "$CLEAN_ROOT/metadata/combined.csv" \
  --feature-file "$CLEAN_ROOT/data/features.csv" \
  --split-manifest "$WORK_ROOT/splits/split_manifest.csv" \
  --output-dir "$WORK_ROOT/results/p0-clean" \
  --fold 1 \
  --streams j,b --xy-data --adaptive --attention --edges coord \
  --epochs 20 --batch-size 32 --num-workers 2 --devices 1
```

With no robustness flag, `train.py` delegates to the unchanged legacy dataset
loader.

## 5. Train P1/P2 with on-the-fly train-only corruption

P1 samples one local family (`gaussian`, `spike`, or `missing`) for 50% of
`inner_train` samples. `inner_dev` remains clean for checkpoint selection.

```bash
!PYTHONPATH=/kaggle/working/infant-aagcn python bin/train.py \
  --data-dir "$CLEAN_ROOT/data/streams_2d/combined" \
  --age-file "$CLEAN_ROOT/metadata/combined.csv" \
  --feature-file "$CLEAN_ROOT/data/features.csv" \
  --split-manifest "$WORK_ROOT/splits/split_manifest.csv" \
  --output-dir "$WORK_ROOT/results/p1-local" \
  --fold 1 \
  --streams j,b --xy-data --adaptive --attention --edges coord \
  --robustness-family-set local \
  --robustness-probability 0.50 \
  --robustness-mild-probability 0.70 \
  --robustness-seed 42 \
  --data-fps 30 \
  --epochs 20 --batch-size 32 --num-workers 2 --devices 1
```

For P2, change the output directory and use:

```text
--robustness-family-set all
```

P3 mixed corruption is intentionally outside this first Kaggle pilot.

## 6. Train on one saved fixed scenario

This is useful for a controlled scenario experiment, not as a replacement for
the more diverse on-the-fly P1/P2 augmentation:

```bash
%env SCENARIO=/kaggle/working/infant-robustness/scenarios/gaussian-moderate-r0-s42

!PYTHONPATH=/kaggle/working/infant-aagcn python bin/train.py \
  --data-dir "$SCENARIO/data/streams_2d/combined" \
  --age-file "$SCENARIO/metadata/combined.csv" \
  --feature-file "$SCENARIO/data/features.csv" \
  --split-manifest "$WORK_ROOT/splits/split_manifest.csv" \
  --output-dir "$WORK_ROOT/results/fixed-gaussian-moderate" \
  --fold 1 \
  --streams j,b --xy-data --adaptive --attention --edges coord \
  --epochs 20 --batch-size 32 --num-workers 2 --devices 1
```

## 7. Inspect artifacts before the Kaggle session ends

```bash
!du -sh /kaggle/working/infant-robustness/*
!find /kaggle/working/infant-robustness -maxdepth 3 -type f | head -100
```

Do not write generated artifacts back into `/kaggle/input`; that mount is
read-only. Publishing the working directory as a versioned Kaggle Dataset is a
separate step.
