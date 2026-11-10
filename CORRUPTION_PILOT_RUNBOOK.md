# Corruption pilot runbook trên Kaggle

Runbook này dùng để chọn corruption policy trên `inner_train/inner_dev` trước
khi mở `outer_test`. Mọi lệnh P0 và P1 phải dùng cùng dataset, split manifest,
kiến trúc, training seed và training budget.

## 0. Thiết lập chung và đóng băng split

Trong notebook Kaggle:

```python
%env REPO_ROOT=/kaggle/working/infant-aagcn
%env DATASET_ROOT=/kaggle/input/datasets/finsosweet/datasetka
%env AGE_FILE=/kaggle/input/datasets/finsosweet/datasetka/metadata/metadata/combined.csv
%env FEATURE_FILE=/kaggle/input/datasets/finsosweet/datasetka/data/data/features.csv
%env STREAM_DIR=/kaggle/input/datasets/finsosweet/datasetka/data/data/streams_2d/combined
%env SPLIT_ROOT=/kaggle/input/infant-frozen-splits
%env WORK_ROOT=/kaggle/working/infant-pilot
```

`DatasetKA` có thêm một cấp folder lặp lại (`data/data` và
`metadata/metadata`), nên không ghép đường dẫn theo mẫu `root/data` cũ. Xác
minh mount trước khi chạy:

```python
import os
from pathlib import Path

assert Path(os.environ["AGE_FILE"]).is_file()
assert Path(os.environ["FEATURE_FILE"]).is_file()
assert list(Path(os.environ["STREAM_DIR"]).glob("*.feather"))
print("DatasetKA paths are valid")
```

Nếu chưa có frozen split, chỉ tạo **một lần** rồi publish cả
`split_manifest.csv` và `split_config.json` thành một Kaggle Dataset:

```bash
!PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/bin/make_robustness_splits.py" \
  --age-file "$AGE_FILE" \
  --output-dir "$WORK_ROOT/frozen-splits" \
  --k-folds 10 --inner-dev-fraction 0.20 --age-bins 5 --seed 42
```

`split_manifest.csv` là file chứa membership thật của từng subject.
`split_config.json` chứa cấu hình và SHA-256 để phát hiện split bị thay đổi.
Không sinh lại hai file này trong notebook P0 hoặc P1.

## 1. Audit 100 segment thật

Audit chỉ đọc `inner_train` của fold 1. Nó không đọc `outer_test` và không cần
GPU.

```bash
!PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/bin/audit_corruption.py" \
  --data-dir "$STREAM_DIR" \
  --feature-file "$FEATURE_FILE" \
  --split-manifest "$SPLIT_ROOT/split_manifest.csv" \
  --output-dir "$WORK_ROOT/audit-fold-1" \
  --fold 1 --role inner_train --sample-size 100 \
  --families gaussian,spike,missing,scale,rotation \
  --severities mild,moderate --replicates 1 \
  --sample-seed 42 --corruption-seed 42 --fps 30 --xy-data
```

### Output và ý nghĩa

- `sampled_segments.csv`: 100 segment đã audit. Lần audit policy mới, truyền
  file này qua `--sample-manifest` để dùng đúng cùng tập mẫu.
- `per_segment_metrics.csv`: một hàng cho mỗi
  `segment × family × severity × replicate`. Dùng để tìm outlier và segment bị
  phá đặc biệt mạnh.
- `summary_metrics.csv`: median và p95 theo family/severity. Đây là bảng chính
  để chọn severity.
- `audit_config.json`: tham số corruption, seed và hash của split manifest.
  Đây là provenance của lần audit.

Các metric chính:

| Metric | Cách đọc |
|---|---|
| `displacement_p95_body_scale` | P95 độ lệch joint chia cho body scale `L`; `0.05` nghĩa là lệch bằng 5% kích thước cơ thể tham chiếu. |
| `j_motion_energy_ratio` | Năng lượng first-difference sau/trước corruption. Gần `1` là giữ biên độ động học; rất lớn là thêm high-frequency motion; rất nhỏ là làm phẳng chuyển động. |
| `j_velocity_correlation` | Tương quan velocity sạch và corrupted. Gần `1` là giữ dạng chuyển động; gần `0` nghĩa là temporal pattern gần như mất. |
| `b_motion_energy_ratio`, `b_velocity_correlation` | Cùng hai phép đo trên bone stream `B`; có thể xấu hơn `J` vì phép trừ child-parent lan truyền nhiễu. |
| `bone_length_relative_error_p95` | P95 sai lệch chiều dài xương tương đối. Rotation nên gần `0`; scale thay đổi chiều dài đồng đều theo scale factor. |
| `reconstruction_rmse_body_scale` | Chỉ cho missing: RMSE sau linear recovery tại các tọa độ đã cố tình che, chuẩn hóa theo `L`. |
| `reconstruction_displacement_p95_body_scale` | Chỉ cho missing: lỗi phục hồi p95 tại các joint-frame bị che. |

Các kiểm tra logic trước khi train:

- Rotation phải có bone-length error gần 0 và motion-energy ratio gần 1.
- Scale phải có velocity correlation gần 1; energy thay đổi theo bình phương
  scale factor.
- Gaussian/Spike không được làm motion-energy tăng nhiều bậc độ lớn trên phần
  lớn segment hoặc kéo velocity correlation về gần 0.
- `moderate` phải xấu hơn `mild` theo cách có kiểm soát.
- Gate pilot đã đề xuất cho local displacement p95 là `<= 0.05 L`; spike
  fraction `<= 0.025`; missing fraction `<= 0.12`. Motion-energy/correlation
  chưa có ngưỡng infant-specific, nên phải khóa ngưỡng từ audit trước khi mở
  outer test.

Giảm `robustness-probability` chỉ giảm số lần một corruption xuất hiện khi
train; nó không sửa một corruption vốn đã phi thực tế trên từng sample. Nếu
Gaussian/Spike quá mạnh, cần giảm alpha/magnitude hoặc dùng noise có temporal
correlation, sau đó chạy lại audit và tạo policy version mới.

## 2. Train P0 sạch trên fold 1

```bash
!PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/bin/train.py" \
  --data-dir "$STREAM_DIR" \
  --age-file "$AGE_FILE" \
  --feature-file "$FEATURE_FILE" \
  --split-manifest "$SPLIT_ROOT/split_manifest.csv" \
  --output-dir "$WORK_ROOT/results/p0-clean" \
  --fold 1 --k-folds 10 --streams j,b --xy-data \
  --adaptive --attention --edges coord \
  --epochs 20 --batch-size 32 --learning-rate 0.01 \
  --hidden-dim 256 --kt 13 --num-workers 2 --devices 1 \
  --train-seed 42
```

### Output và ý nghĩa

- `fold_1/best_model.ckpt`: checkpoint có clean `inner_dev val_loss` thấp nhất.
- `fold_1/metadata.json`: args, loss theo epoch, subject/segment membership và
  clean inner-dev metrics.
- `train_predictions.npy`, `val_predictions.npy`: prediction sạch dùng để
  kiểm tra scatter/error distribution.
- `last.ckpt` được cập nhật trong lúc train để có recovery artifact nếu output
  của phiên bị gián đoạn còn được Kaggle lưu. CLI hiện chưa tự động resume từ
  file này; sau một run hoàn tất, script chỉ giữ `best_model.ckpt`.

P0 trả lời: **với protocol inner split mới, model sạch có học được không?** Nó
không phải kết quả final outer-test và không nên so trực tiếp với baseline
legacy dùng nhiều training subject hơn.

Dấu hiệu cần dừng: train loss không giảm, prediction gần như hằng số, MAE vô
lý, hoặc best epoch xuất hiện ngay đầu rồi suy giảm mạnh. Phải sửa P0 trước khi
đánh giá corruption.

## 3. Đánh giá P0 trên clean và fixed-corrupted inner-dev

```bash
!PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/bin/evaluate_robustness.py" \
  --model-dir "$WORK_ROOT/results/p0-clean" \
  --data-dir "$STREAM_DIR" \
  --feature-file "$FEATURE_FILE" \
  --split-manifest "$SPLIT_ROOT/split_manifest.csv" \
  --output-dir "$WORK_ROOT/eval/p0-fold-1" \
  --fold 1 --families gaussian,spike,missing \
  --severities mild --replicates 3 --corruption-seed 42 \
  --batch-size 32 --num-workers 2 --device auto
```

Script cố định chỉ đọc `inner_dev`. Cùng family/seed/replicate tạo cùng mask và
noise latent khi đánh giá P0 và P1.

- `scenario_metrics.csv`: MAE/RMSE/bias cho clean và từng corrupted scenario,
  cùng `prediction_drift_mae_vs_clean`.
- `predictions.csv`: prediction từng segment để làm paired comparison.
- `evaluation_config.json`: hash checkpoint, hash split và xác nhận
  `outer_test_accessed: false`.

P0 corrupted evaluation trả lời: **model sạch nhạy với từng loại lỗi đến mức
nào?** `mae_delta_vs_clean` đo mất hiệu năng do corruption;
`prediction_drift_mae_vs_clean` đo prediction thay đổi bao nhiêu dù nhãn không
đổi.

## 4. Train P1 nhẹ: 20% corrupted, chỉ mild

```bash
!PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/bin/train.py" \
  --data-dir "$STREAM_DIR" \
  --age-file "$AGE_FILE" \
  --feature-file "$FEATURE_FILE" \
  --split-manifest "$SPLIT_ROOT/split_manifest.csv" \
  --output-dir "$WORK_ROOT/results/p1-local-light" \
  --fold 1 --k-folds 10 --streams j,b --xy-data \
  --adaptive --attention --edges coord \
  --robustness-family-set local \
  --robustness-probability 0.20 \
  --robustness-mild-probability 1.0 \
  --robustness-seed 42 --data-fps 30 \
  --epochs 20 --batch-size 32 --learning-rate 0.01 \
  --hidden-dim 256 --kt 13 --num-workers 2 --devices 1 \
  --train-seed 42
```

P0 và P1 dùng cùng `train-seed`, nên fold tương ứng bắt đầu từ cùng model
initialization. P1 chỉ khác ở train-time corruption.

P1 trả lời: **một lượng augmentation nhẹ có cải thiện robustness mà không làm
hỏng clean performance hay không?**

## 5. Đánh giá P1 trên đúng các scenario của P0

Chạy lại lệnh ở bước 3, chỉ đổi:

```text
--model-dir  .../results/p1-local-light
--output-dir .../eval/p1-fold-1
```

Các tham số family, severity, replicate và seed phải giữ nguyên.

So sánh nhanh:

```python
import pandas as pd

p0 = pd.read_csv('/kaggle/working/infant-pilot/eval/p0-fold-1/scenario_metrics.csv')
p1 = pd.read_csv('/kaggle/working/infant-pilot/eval/p1-fold-1/scenario_metrics.csv')
comparison = p0.merge(p1, on=['fold', 'scenario', 'family', 'severity', 'replicate'],
                      suffixes=('_p0', '_p1'))
comparison['p1_mae_change_vs_p0'] = comparison.mae_p1 - comparison.mae_p0
comparison['p1_relative_mae_change_vs_p0'] = (
    comparison.p1_mae_change_vs_p0 / comparison.mae_p0
)
comparison['robustness_gain'] = comparison.mae_p0 - comparison.mae_p1
comparison.to_csv('/kaggle/working/infant-pilot/eval/p0-vs-p1.csv', index=False)
comparison
```

Cách kết luận:

- Dòng `clean`: `p1_relative_mae_change_vs_p0 <= 0.05` là clean-performance
  guardrail pilot. P1 không nên làm clean MAE tăng quá 5% tương đối.
- Dòng corrupted: `robustness_gain > 0` nghĩa là P1 tốt hơn P0 trên scenario
  đó.
- Chỉ chấp nhận P1 nếu qua clean guardrail và cải thiện có tính nhất quán giữa
  replicate/segment; không chọn chỉ vì một scenario may mắn.
- Nếu clean tốt nhưng corrupted không cải thiện, augmentation không mang lại
  robustness cần thiết.
- Nếu corrupted tốt nhưng clean giảm quá 5%, policy đang đánh đổi quá nhiều.

## 6. Điều chỉnh và khóa policy

Chỉ dùng audit và `inner_dev` để điều chỉnh. Mỗi lần đổi alpha/rate/magnitude,
temporal correlation, family weights hoặc recovery phải:

1. Tạo một policy version mới.
2. Chạy lại audit trên đúng `sampled_segments.csv` bằng
   `--sample-manifest <audit-cũ>/sampled_segments.csv`.
3. Chạy lại P1 fold 1 với cùng P0 budget.
4. Đánh giá bằng đúng fixed scenarios.

Khi policy đạt gate, lưu tối thiểu:

- `split_manifest.csv` và SHA-256;
- `audit_config.json`, `summary_metrics.csv`;
- train args/seeds và code commit;
- evaluation config và scenario seeds;
- quy tắc aggregation và clean-MAE guardrail.

Sau thời điểm khóa, không thay policy dựa trên bất kỳ `outer_test` result nào.
Script `evaluate_robustness.py` cố ý chỉ hỗ trợ `inner_dev`; final outer-fold
evaluation nên là một command riêng chỉ được chạy sau khi policy lock hoàn
tất.

## 7. Khi nào mới chạy đủ 10 fold?

Chỉ tiếp tục khi cả ba điều kiện đều đạt:

1. Audit không cho thấy temporal dynamics bị Gaussian/Spike thống trị.
2. P0 fold 1 học ổn định trong protocol mới.
3. P1 fold 1 qua clean-MAE guardrail và cải thiện fixed-corrupted inner-dev.

Sau đó chạy từng fold thành một Kaggle version riêng cho P0/P1. Không chạy cả
10 fold trong một phiên 12 giờ.
