# Train the 2D JB-AAGCN model on Kaggle

This workflow uses the author's preprocessed 2D data and Kaggle's preinstalled
CUDA-enabled PyTorch. Do not install the root `requirements.txt` on Kaggle: it
targets PyTorch 1.13 and CUDA 11.6.

## 1. Required Kaggle dataset

Attach a Kaggle dataset with this structure:

```text
metadata/combined.csv
data/features.csv
data/streams_2d/combined/*.feather
```

The public preprocessed archive already has this structure. Raw CSV keypoints
cannot be passed directly to `bin/train.py`.

## 2. Notebook setup

Enable a GPU and Internet in the Kaggle notebook settings, then run:

```bash
!git clone https://github.com/FinnNguyenn/infant-aagcn-kaggle-2d.git /kaggle/working/infant-aagcn
%cd /kaggle/working/infant-aagcn
%pip install -q -r requirements-kaggle.txt
```

Find the mounted data path:

```bash
!find /kaggle/input -maxdepth 6 -type f | head -100
```

The examples below assume the dataset is mounted at
`/kaggle/input/infant-aagcn-2d-data`. Change it when the Kaggle dataset slug is
different.

## 3. Validate the paths

```python
from pathlib import Path

root = Path('/kaggle/input/infant-aagcn-2d-data')
age_file = root / 'metadata' / 'combined.csv'
feature_file = root / 'data' / 'features.csv'
stream_dir = root / 'data' / 'streams_2d' / 'combined'

assert age_file.is_file(), age_file
assert feature_file.is_file(), feature_file
assert list(stream_dir.glob('*.feather')), stream_dir
print('2D dataset is ready')
```

## 4. Smoke test

```bash
!PYTHONPATH=/kaggle/working/infant-aagcn python bin/train.py \
  --data-dir /kaggle/input/infant-aagcn-2d-data/data/streams_2d/combined \
  --age-file /kaggle/input/infant-aagcn-2d-data/metadata/combined.csv \
  --feature-file /kaggle/input/infant-aagcn-2d-data/data/features.csv \
  --output-dir /kaggle/working/results/smoke-2d \
  --streams j,b \
  --xy-data \
  --adaptive \
  --attention \
  --edges coord \
  --k-folds 2 \
  --epochs 1 \
  --batch-size 4 \
  --hidden-dim 32 \
  --num-workers 2 \
  --devices 1
```

## 5. Full 2D training

The following reproduces the repository's 2D experiment configuration:

```bash
!PYTHONPATH=/kaggle/working/infant-aagcn python bin/train.py \
  --data-dir /kaggle/input/infant-aagcn-2d-data/data/streams_2d/combined \
  --age-file /kaggle/input/infant-aagcn-2d-data/metadata/combined.csv \
  --feature-file /kaggle/input/infant-aagcn-2d-data/data/features.csv \
  --output-dir /kaggle/working/results/jb-aagcn-coord-xy \
  --streams j,b \
  --xy-data \
  --adaptive \
  --attention \
  --edges coord \
  --k-folds 10 \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.01 \
  --hidden-dim 256 \
  --kt 13 \
  --num-workers 2 \
  --devices 1
```

If GPU memory is exhausted, reduce `--batch-size` to 16 or 8.

To make a Kaggle run resilient to runtime limits, add `--fold N` to train only
fold `N`, where `N` is from 1 through 10. Reuse the same output directory for
all folds. Each completed fold writes:

```text
results/jb-aagcn-coord-xy/fold_N/best_model.ckpt
results/jb-aagcn-coord-xy/fold_N/metadata.json
```

Example for fold 1:

```bash
!PYTHONPATH=/kaggle/working/infant-aagcn python bin/train.py \
  --data-dir /kaggle/input/infant-aagcn-2d-data/data/streams_2d/combined \
  --age-file /kaggle/input/infant-aagcn-2d-data/metadata/combined.csv \
  --feature-file /kaggle/input/infant-aagcn-2d-data/data/features.csv \
  --output-dir /kaggle/working/results/jb-aagcn-coord-xy \
  --streams j,b --xy-data --adaptive --attention --edges coord \
  --k-folds 10 --fold 1 --epochs 20 --batch-size 32 \
  --num-workers 2 --devices 1
```
