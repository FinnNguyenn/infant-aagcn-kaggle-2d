# Codex Implementation Specification — Controlled Infant Skeleton Corruption Benchmark

**Ngày:** 2026-11-04 · **Bản sửa đổi:** v4 (locked inner/outer protocol, consistent checkpointing, separated repair/preprocessing, non-destructive integration)
**Đối tượng:** Codex triển khai Python module tạo nhiễu trên skeleton 2D/3D để đánh giá preprocessing và **Kinetic Age regression** (Holmberg / JB-AAGCN).
**Mục đích:** tạo các lỗi quan sát có kiểm soát, có cơ sở toán học và liên quan đến video infant; **không** cố ý hủy chuyển động, **không** làm nhiễu nhãn tuổi.
**Phạm vi đúng 5 families:** (1) localization jitter, (2) spike outliers, (3) missing/occlusion (random + temporal gaps), (4) global skeleton rescaling, (5) global skeleton rotation.

> **CODEx — Start here:** Đọc kỹ spec này và inspect repo hiện có trước khi chỉnh sửa. Chỉ thêm module corruption + tests + config + CLI/example tích hợp theo cấu trúc repo đang có. **Không tự ý sửa thuật toán pipeline 4554/3123, kiến trúc model, outer folds, dữ liệu gốc, checkpoints hoặc downstream labels.** ĐƯỢC phép tạo manifest inner_train/inner_dev bên trong outer_train cho protocol mới, không thay đổi legacy replication. Được phép viết wrapper imputation/recovery, dataset adapter và training augmentation hook mới; không sửa hành vi cũ khi tắt feature flag. Nếu repo chưa có phần tương ứng, tạo các hàm độc lập, không đoán đường dẫn. Ưu tiên 1 chức năng = 1 commit; test qua môi trường hiện có (nếu là `gm-pps-py310` thì dùng môi trường đó). Không commit nếu tests fail.

## 0. Triển khai tối thiểu an toàn (v4) — Codex execution order

1. **Inspect only** the live repo (`bin/train.py`, `modules/data.py`, model, graph constants, split code); confirm assumptions in this document. Không giả định line numbers trong review là current truth.
2. **Không thay đổi đường chạy baseline**. Tạo `robustness.enabled` default `false`. Khi false, loader/train/model chạy y hệt legacy; unit-test parity.
3. Tạo outer/inner subject-level manifests và khóa seed, model budget, checkpoint MSE **trước** P0. Không lấy kết quả outer fold để tuning.
4. Core: 5 single families + deterministic masks/seeds + NaN recovery + JB-only + `scenario_bank` for development; test toán học và no-NaN.
5. Train P0 và thử `inner_dev` sensitivity; train P1/P2 với on-the-fly corruption; P3 mixed chỉ khi core hoàn chỉnh.
6. Chỉ thêm `epoch_bank` khi đo được bottleneck; không cần `preprocessing_cache` giai đoạn đầu. 1 Feather = 1 sample; augment in `__getitem__` với label lookup theo sample ID, không nhân thêm `augment_name`.
7. Commit theo chức năng, mỗi commit test pass; output gồm split manifests, audit logs, config hashes, artifacts đường dẫn rõ ràng, README run commands; **không tự chạy final outer test trong lúc phát triển**.

## 1. Cơ sở bài toán và các giả định

Cho chuỗi skeleton tham chiếu đã công bố:

\[
X\in\mathbb R^{T\times J\times C},\quad C\in\{2,3\},
\]

với `T` frames, `J` joints, `C` tọa độ. `X` là **processed reference**, không phải ground-truth pose hoàn hảo. Sinh `X_corrupt` từ `X` qua toán tử corruption `C_θ`; giữ `X` nguyên vẹn để đo reconstruction và age-prediction robustness.

- Có thể áp dụng trên Holmberg Feather bằng cách **chỉ lấy stream J**; `B` **phải tái tính từ J sau corruption/recovery**; `V/A` nằm ngoài pilot JB-AAGCN; không chèn noise độc lập vào các stream sẽ gây mâu thuẫn hình học.
- Đảm bảo cùng subject-ID không xuất hiện ở cả train và test. Sinh corruption **sau subject-level split**, seed được xác định từ sample ID + experiment seed (không dùng `hash()` của Python vì không ổn định giữa tiến trình).
- Hai kiểu thí nghiệm khác nhau: **evaluation-only stress testing** (giữ pretrained model) và **training-time augmentation** (chỉ train data; retrain mô hình). Không trộn các kết luận.
- Cả 2D và 3D cần cùng semantics. Không ép 2D thành 3D giả. Tọa độ có thể là pixels, meters, hoặc unitless; không áp ngưỡng tuyệt đối như `σ=0.1` mà chưa biết đơn vị.
- Holmberg đã center, rotate và có thể lặp tuần hoàn segment để đạt 600 frames. **Không được giả định 600 frame là một đoạn quan sát liên tục ban đầu.** Nếu có original-length/loop-boundary metadata thì chỉ corrupt và xử lý trên đoạn thực, rồi áp dụng repetition tương ứng. Nếu không có: đánh dấu `post_segmentation_corruption=true`, không smoothing/filter qua seam chưa xác định mà không ghi hạn chế; không tự nhận đã tái tạo raw-data noise.

## 2. Các nguyên tắc không được vi phạm

1. **Bảo toàn nhãn và temporal identity:** Không đổi `age`, `subject_id`, số frames, joint order, FPS hay thứ tự thời gian. Trong mọi sample, phép scale/rotation được **giữ cố định cho toàn bộ segment**, không quay/scale mỗi frame độc lập.
2. **Bảo toàn cấu trúc có ý nghĩa:** Với corruption hình học, áp dụng **cùng ma trận/scale lên mọi joints**: giữ bone length ratios, relative angles và inter-joint topology. Với noise cục bộ, làm hỏng **một phần nhỏ**; không thay đổi đồng thời toàn bộ joints với biên độ quá lớn.
3. **Bảo toàn nội dung chuyển động:** Không time reversal, random frame permutation, frame deletion, temporal resampling hay smoothing bên trong corruption module. Không giả lập các pattern bất thường về thần kinh rồi gán cho chúng nhãn bệnh.
4. **Không tạo điểm 0 giả:** Missing được biểu diễn bằng `NaN` và `observed_mask=False`, **không** set tọa độ `(0,0[,0])` (vì gốc là tọa độ hợp lệ). **AAGCN không được phép nhận NaN:** bắt buộc đưa qua recovery contract ở §6.1 trước khi recompute B (pilot JB-AAGCN) và inference/training; riêng bài test reconstruction có thể giữ NaN ở output corruption thô.
5. **Không nhân tạo confidence như ground-truth:** Nếu input có confidence thật thì đưa `confidence_corrupt` ra riêng. Missing → confidence=0; jitter/spike confidence giảm chỉ khi có một giả định rõ được bật trong config, mặc định giữ nguyên confidence và ghi rõ synthetic corruption không tái tạo reliability calibration của pose estimator.
6. **Không ngầm làm sạch reference:** `X` bất biến; nếu `X` có NaN sẵn, corruption chỉ áp vào vị trí observed, không khôi phục bằng phép tạo nhiễu. Kiểm tra `np.isfinite` đúng ở các vị trí observed.
7. **Ưu tiên mức nhẹ/vừa:** Severe là stress-test phụ, không phải điều kiện mặc định. Không lựa chọn severity bằng cách nhìn vào test labels.
8. **Biến đổi không được out-of-frame một cách vô lý:** Với 2D pixel-space, kiểm tra bounding box sau scale/rotate. Nếu vượt ảnh ở chế độ `strict_fov`, resample hệ số/góc trong số lần thử giới hạn, nếu vẫn không hợp lệ thì skip/report — **không clip từng joint** vì clip tạo biến dạng động học. Với Holmberg 3D centered coordinates không có ảnh gốc/FOV, đừng áp giả định pixel-bound.

## 3. Quy ước scale tham chiếu và tâm biến đổi

**`body_scale L` (khóa công thức và thứ tự fallback):** Với frame `t`, tính `midhip_t = (left_hip_t + right_hip_t)/2` và `length_t = ||neck_t - midhip_t||_2`. Lấy `L = median(length_t)` trên các frame **đủ các khớp hợp lệ, finite, và chiều dài > epsilon**. Các tham số pilot `min_valid_scale_frames=30`, `min_valid_scale_fraction=0.05`, `epsilon=1e-6` trong đơn vị tọa độ phải được khóa trước evaluation (có thể điều chỉnh nếu coordinate units khác; không chọn theo test). Nếu không đủ frame, thử đúng theo thứ tự: (1) neck–midhip; (2) left–right shoulder width; (3) left–right hip width. Mỗi fallback cũng dùng median trên valid frames với cùng criteria. Nếu vẫn không đủ, `status=skipped` và `reason=invalid_body_scale`; **không** âm thầm dùng `L=1`. Tính một lần từ **reference clean X**, dùng cùng `L` và `scale_method` cho mọi severity và replicate của sample đó. Ghi `L`, `scale_method`, `valid_scale_frames`, units và frame eligibility trong metadata. Không trộn các method giữa severity của cùng sample. Holmberg centered coordinates không bảo đảm `L` tương ứng độ dài cơ thể thật.

**Anchor / pivot:** xác định một tâm **DUY NHẤT cho toàn bộ segment** (không phải `(0,0)` theo mặc định), dùng `pivot_mode`:

- `median_neck`: median theo thời gian của neck joint từ reference;
- `median_midhip`: median theo thời gian của điểm giữa left/right hip;
- `median_body_center`: median theo thời gian của trung tâm các joints hợp lệ;
- `image_center`: `(width/2,height/2)` **chỉ** khi có dimensions trong pixel-space; hợp lý để biến đổi quanh tâm khung hình nhưng có thể khiến skeleton không ở giữa nếu ban đầu nó lệch tâm.

**Default theo dataset:** Holmberg Feather đã center neck ⇒ `median_neck` (`recenter_after_transform: false`); video raw/pixel-space ⇒ `median_midhip` nếu landmark đủ, fallback `median_neck`. Không fallback âm thầm giữa các pivot khi đánh giá cùng benchmark; log lựa chọn chính xác. Giữ anchor nguyên trong tất cả frames của một segment. **Đây là điều kiện user yêu cầu để skeleton không bị co về góc ảnh khi thu nhỏ.** Với Holmberg, pivot gần zero để bảo toàn neck-at-origin. Nếu cố tình dùng midhip trên Holmberg, gọi thí nghiệm `scale_plus_translation` / `rotation_plus_translation` hoặc re-center neck và báo cáo việc re-centering.

Với anchor cố định `c∈R^C`, công thức affine đúng:

\[
\boxed{\widetilde X_{t,j}=c+s\,(X_{t,j}-c)}
\]

Không được dùng `X_corrupt = s*X` trừ khi chủ đích scale quanh origin và origin đúng là pivot. Kiểm tra centroid theo anchor: với `s<1`, tất cả joints đi gần `c` hơn, và `c` **không đổi**. Để mô phỏng thay đổi camera distance, cần lưu ý phép scaling 2D là xấp xỉ của zoom/pinhole trong điều kiện giới hạn, không phải biến đổi hình chiếu 3D chính xác.

## 4. Family 1 — Gaussian localization jitter

**Failure mode thực tế:** sai số định vị keypoint giữa các frames do pose estimator (ánh sáng, da/quần áo, landmark ambiguity). Đây là baseline measurement-error, **không tuyên bố residual thật là Gaussian i.i.d.**

\[
\widetilde X_{t,j}=X_{t,j}+\epsilon_{t,j},\quad
\epsilon_{t,j}\overset{iid}{\sim}\mathcal N(0,(\alpha L)^2I_C).
\]

- `α` là std tương đối với `L`, default pilot: `mild=0.005`, `moderate=0.015`, `severe=0.03`.
- Chỉ add noise tại observed joints; không tạo corruption cho NaN. Toàn bộ tọa độ `C` của một joint được perturb bằng vector đồng nhất về thời điểm lựa chọn.
- Thêm option `affected_joint_rate` (mặc định 1 với Gaussian test; có thể thấp hơn để mô phỏng chỉ một số landmarks khó quan sát). Không dùng temporal smoothing trong generator.
- Optional extension (không bắt buộc): AR(1) temporally correlated errors `ε_t=ρ ε_{t−1}+η_t`, `η_t∼N(0,(1−ρ²)(α L)²I)`; nếu triển khai phải tách `gaussian_iid` và `gaussian_ar1` trong metadata và test stationary variance.

## 5. Family 2 — Sparse spike outliers

**Failure mode:** keypoint bị nhận nhầm đột ngột sang vị trí khác khi bị che hoặc chuyển động nhanh. **Không gọi random spike là left/right swap**; swap là một corruption khác, nằm ngoài default. Contamination model:

\[
\widetilde X_{t,j}=X_{t,j}+Z_{t,j} D_{t,j},\quad
Z_{t,j}\sim\mathrm{Bernoulli}(p),
\]

\[
D_{t,j}=m_{t,j}u_{t,j},\quad
\|u_{t,j}\|_2=1,\quad m_{t,j}\sim\mathcal U(aL,bL).
\]

- Pilot `p`: mild 0.005, moderate 0.02, severe 0.05 theo joint-frame.
- Pilot displacement `[a,b]`: mild `[0.05,0.15]`, moderate `[0.10,0.30]`, severe `[0.20,0.50]` (đơn vị L). **Đây là đề xuất, không phải tham số chuẩn infant.**
- Mask xuất `corruption_mask[t,j]=True` đúng ở các joint-frame bị tác động.
- Giới hạn outlier burst: default single-frame; optional burst 2–3 frames phải có metadata riêng. Tránh lấy toàn bộ frame làm spike đồng thời.
- Không xóa joint hoặc clip tọa độ bị sai sau khi sinh; việc repair thuộc preprocessing.

## 6. Family 3 — Missing joints / occlusion (hai submodes, cùng một family)

**Failure mode:** keypoint detector mất landmark hoặc limb bị che liên tiếp khi tay/chân áp sát thân, nằm dưới quần áo, v.v. Observation model:

\[
M_{t,j}\in\{0,1\},\quad \widetilde X_{t,j}=\begin{cases}X_{t,j}, & M_{t,j}=1\\\mathrm{NaN}, & M_{t,j}=0.\end{cases}
\]

### 3A. `random_missing` — MCAR simulation

- Mặc định dùng **exact-count masking** trên các vị trí quan sát được: sample một ma trận uniform cố định rồi chọn `round(q*N_observed)` scores thấp nhất. Việc này giữ đúng rate và có thể lồng nhau giữa các severity. Nếu bật Bernoulli thay thế, log rate thực tế, không reject ngẫu nhiên chỉ vì vượt kỳ vọng.
- Pilot `q`: mild 0.03, moderate 0.08, severe 0.15 (cố ý nhẹ hơn nhiều benchmark adult HAR).
- Đây là mô hình MCAR do nhà nghiên cứu chỉ định, **không** có nghĩa occlusion thật của infant là MCAR.

### 3B. `temporal_gap` — temporally persistent occlusion

- Chọn 1 hoặc số nhỏ joints ngoại biên, chọn đoạn liên tiếp dài `d` rồi đánh dấu mất. Pilot `d` (ở `fps=25`): mild 2–3 frames, moderate 4–7, severe 8–12. Nếu FPS khác, quy đổi **theo giây** thay vì áp cứng số frame; lưu `gap_duration_seconds`.
- Mặc định tránh chọn **mọi joints trên cả một khoảng**; tránh mất đồng thời cả hai tay/chân. Có option chọn cùng một limb (wrists + elbows hoặc ankles + knees) để stress test riêng, không nằm trong default.
- Dùng `NaN` + `observed_mask` **ở giai đoạn corruption**; không đưa trực tiếp sang AAGCN. Chạy recovery thống nhất theo §6.1 ở mọi branch cần model.
- Cần test `mask` có đúng gap length, không vượt bounds, random missing rate được hiện thực đúng; nếu reference đã missing thì `new_corruption_mask = newly_removed_only`.

**Cơ sở toán học:** Rubin (1976) phân biệt MCAR/MAR/MNAR; ở đây chỉ định missing mechanisms cho simulation. Structured occlusion được dùng trong các nghiên cứu masked skeleton và occlusion-aware activity recognition. Không gắn nhãn MAR/MNAR nếu chưa định nghĩa xác suất missing có điều kiện cụ thể.

### 6.1. Mandatory missing recovery contract trước JB-AAGCN

**Bắt buộc áp dụng một đường xử lý chung cho tất cả benchmark so sánh models:**

```text
Clean J → synthetic corruption → missing recovery (nếu cần)
        → optional preprocessing candidate → recompute B (và V/A nếu sử dụng)
        → verify finite tensor → JB-AAGCN
```

- Default model-path `recovery_method: linear`; chỉ nội suy trên gap **nội bộ** của từng joint và coordinate. `PCHIP` là **candidate method** để so sánh ở nhánh preprocessing, không được thay lẫn vào fixed-model robustness nếu không ghi rõ.
- Gap đầu/cuối: điền giá trị quan sát gần nhất (boundary nearest fill). **Không** kéo nối qua seam của segment lặp khi biết seam.
- Joint trajectory mất toàn bộ hoặc thiếu reference đáng tin: `reject_sample` + log `all_missing_joint`; không điền zeros hay trung bình các khớp khác theo mặc định. Nếu có reconstruction chuyên biệt, coi là experiment khác.
- Nếu không có newly missing thì pass-through; `NaN` có từ reference phải được xử lý theo documented data-quality policy và báo cáo riêng.
- Sau recovery: kiểm tra toàn bộ `J`, `B` và các stream thực dùng đều finite; missing `observed_mask` được lưu để đánh giá reconstruction trên corrupted sites.
- So sánh các preprocessing methods phải dùng cùng corrupted input/mask và cùng baseline recovery policy (trừ khi chính recovery là biến độc lập của experiment).
- **Không thay đổi model thành mask-aware AAGCN** trong bản này: đây là hướng nghiên cứu riêng, không thể so sánh trực tiếp với checkpoint ban đầu.

## 7. Family 4 — Random GLOBAL skeleton rescaling (pivot-preserving)

**Failure mode gần thực tế:** trẻ to/nhỏ khác nhau, thay đổi zoom/khoảng cách camera và pixel-to-world scale. Đây là **geometric nuisance / domain shift**, không phải noise keypoint; không khẳng định scale không chứa tuổi sinh học.

\[
\boxed{\widetilde X_{t,j}=c+s(X_{t,j}-c),\quad s>0.}
\]

- `s~U(1-β,1+β)`, **một `s` cho cả segment**.
- Pilot: mild `β=0.05`, moderate `β=0.10`, severe `β=0.20`. Default chạy mild và moderate; severe chỉ stress-test.
- `pivot_mode=median_neck` **khi adapter=Holmberg**; với raw/pixel-space thì ưu tiên `median_midhip`. Phải **giữ nguyên pivot**: `T(c)=c`; skeleton thu nhỏ/quay không bị hút về origin hay góc trái trên.
- Đừng scale riêng x và y (anisotropic) trong default: làm biến dạng tỷ lệ xương. Đừng scale từng limb độc lập.
- Giữ tất cả joint-to-joint angle, ratio of bone lengths, hình dạng chuyển động tương đối. Speed/acceleration theo đơn vị tọa độ cũng scale theo s; phải phân tích đúng ý nghĩa khi đánh giá feature error.
- Nếu có `frame_width/frame_height` và pixel-space: chọn pivot và resample s khi làm vượt FOV; không clip.
- **Giới hạn ý nghĩa:** với Holmberg 3D (world coordinates), scale 3D không giống camera lùi/tiến; với Holmberg 2D (3D-projection) cũng không phải camera RGB zoom thực tế. Kết luận là *scale sensitivity* chứ không phải camera-distance robustness đã được kiểm định.

## 8. Family 5 — GLOBAL rotation (pivot-preserving)

**Failure mode gần thực tế:** khác góc đặt camera / orientation của trẻ / sai lệch căn chỉnh hệ quy chiếu. Đây là geometric transformation. Holmberg có rotation normalization trong pipeline; đây là stress test khả năng orientation invariance, không phải keypoint-localization corruption.

**2D:**

\[
\boxed{\widetilde X_{t,j}=c+R(\theta)(X_{t,j}-c)},\qquad
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}.
\]

**3D:**

\[
\widetilde X_{t,j}=c+R(X_{t,j}-c),\quad R\in SO(3).
\]

- Sample `θ~U(-θ_max,θ_max)` **một góc cho cả segment**. Pilot mild 5°, moderate 10°, severe 20°. Đây là pilot bảo thủ, không phải phân phối angle của camera trong thực tế.
- Default `axis='z'` cho cả 2D/3D theo convention x-y trong repo; chỉ thêm x/y khi biết rõ axes vật lý và rotation semantics; không dùng random Euler rotations lớn khiến pose nằm ngoài mặt phẳng quay có ý nghĩa.
- Rotation phải trực giao `RᵀR≈I` và `det(R)≈1`, bảo toàn bone lengths và tương quan khoảng cách giữa joints.
- Center cố định `c` giống phần scale, không xoay quanh origin tùy tiện. **Holmberg adapter mặc định median_neck, không midhip.** Với 2D và FOV, resample góc nếu vượt ảnh; không clip từng joint.
- Với Holmberg đã rotation normalized, experiment này chủ yếu kiểm tra độ bền trước **perturbed alignment**, không đại diện chính xác cho việc quay lại video infant thực tế.

## 9. Giới hạn mức phá: controlled corruption / motion preservation

**Không dùng một hard cap affected-fraction chung cho 5 families.** Quy tắc theo loại corruption; những ngưỡng là pilot, không đại diện thống kê lỗi pose infant. Không dùng rejection sampling khiến phân phối corruption bị đổi ngầm; nếu giới hạn và mức severity xung đột thì phải sửa config trước thí nghiệm, không âm thầm ép mẫu qua.

### 9.1 Family-specific hard gates

| Family | Điều kiện bắt buộc | Không áp dụng |
|---|---|---|
| Gaussian dense | `alpha`, per-site displacement audit, hữu hạn trên observed locations | affected-fraction cap (có thể là 100%) |
| Spike | max spike-rate cấu hình, displacement range, burst length | cap của Gaussian |
| Random missing | exact-count q theo observed sites, missing rate thực tế, điều kiện khả thi sau masking | global `<=0.15` với severe q=0.15 |
| Temporal gap | max gap duration (giây), tổng missing fraction, joint/limb count | dùng q của random missing thay thế |
| Scale | `s>0`, fixed pivot, preserve distance ratios, optional pixel-space FOV | affected-fraction cap |
| Rotation | fixed R, `RᵀR≈I`, `det R≈1`, optional pixel-space FOV | affected-fraction cap |

- Mọi family phải giữ shape `(T,J,C)`, dtype số, joint order, timestamps, subject ID và nhãn tuổi. Missing được phép chứa NaN **ở output corruption** nhưng không ở input model.
- `min_visible_joint_fraction` và `min_usable_frame_fraction` chỉ là các **quality metrics / reject rule** được cấu hình riêng cho stress test; không bí mật loại các trường hợp severe khó. Luôn báo cáo tỷ lệ rejected/accepted/invalid.
- Nếu dùng Bernoulli masking thay exact-count, có thể đặt cap lớn hơn kỳ vọng (ví dụ q=.15, cap=.18) để phát hiện lỗi quá mức, **không** tái lấy mẫu cho đến khi qua cap.
- Với scale/rotation cố định toàn sequence, không tự xem biến thiên vị trí là lỗi xương. Với scaling, toàn bộ bone lengths scale theo s (ratios không đổi); với rotation, bone lengths tuyệt đối không đổi.
- Phép biến đổi vượt FOV: chỉ có ý nghĩa khi input pixel-space có frame dimensions; `strict_fov=true` phải log lý do reject hoặc sample transform theo phân phối được định nghĩa trước. Không clip từng joint.

### 9.2 Soft audit metrics (đo và log, không ép cứng cho từng sample)

- `mean_joint_displacement_over_L = mean(||X_corrupt-X||)/L` trên observed sites.
- `velocity_error_over_L_per_s`, `acceleration_error_over_L_per_s2` (tránh derivatives qua repeated segment seam).
- `bone_length_ratio_error` giữa clean và corruption (zero/near-zero cho geometric transformations; có thể khác 0 cho jitter/spikes).
- `motion_energy_ratio` từ first differences trên **các cặp valid and non-seam frames**; không bắt buộc tỉ lệ gần 1 cho jitter/spikes (vì đó chính là tác động cần đo), mà dùng để phát hiện corruption quá mạnh so với tín hiệu gốc.
- `usable_frame_fraction`, `new_missing_fraction`, `max_gap_s`, số spikes, `s`, `theta_deg`, pivot, FOV violations và `skip_reason`.
- Không dùng age RMSE trên held-out test để quyết định chấp nhận hay từ chối corruption. Severity phải cố định trước test.

### 9.3 Thí nghiệm thăm dò để hiệu chỉnh severity (khuyến nghị)

Trước khi chạy full benchmark, lấy ngẫu nhiên nhưng cố định 10–20 segments, vẽ overlay clean/corrupted và trajectory wrist/ankle, velocity profiles; xác nhận các chuyển động tay/chân còn nhận ra được. Nếu một mức phá làm hủy hoàn toàn nội dung, giảm **mức pilot cho cả benchmark trước khi đánh giá test**, ghi thay đổi vào config và changelog. Không tối ưu từng sample để kết quả đẹp.

## 10. API, config và lưu vết bắt buộc

Codex triển khai interface gần với:

```python
@dataclass(frozen=True)
class CorruptionConfig:
    family: str                     # gaussian | spike | missing | scale | rotation
    severity: str                   # mild | moderate | severe
    seed: int = 42
    pivot_mode: str = "median_neck"  # Holmberg default; raw adapter overrides
    missing_submode: str = "random_missing"  # or temporal_gap
    strict_fov: bool = False
    max_attempts: int = 20
    recenter_after_transform: bool = False


def corrupt_skeleton(
    X: np.ndarray,
    config: CorruptionConfig,
    *,
    joint_names: list[str],
    subject_id: str,
    sample_id: str,
    fps: float | None = None,
    confidence: np.ndarray | None = None,
    frame_size: tuple[int, int] | None = None,
    valid_length: int | None = None,
) -> CorruptionResult: ...
```

`CorruptionResult` có `X_corrupt`, `observed_mask`, `corruption_mask`, `confidence_corrupt` (optional), `metadata` (JSON serializable), `status=accepted|skipped`, `reason`. Có thể điều chỉnh API theo repo, nhưng **không bỏ thông tin metadata / masks**.

Ví dụ YAML:

```yaml
corruption:
  family: scale
  severity: moderate
  seed: 42
  pivot_mode: median_neck
  strict_fov: false
  max_attempts: 20
  scale_beta: 0.10
  theta_max_deg: 10.0
  gaussian_alpha: 0.015
  spike_rate: 0.02
  spike_magnitude_range_L: [0.10, 0.30]
  missing_submode: random_missing
  missing_rate: 0.08
  gap_seconds_range: [0.16, 0.28]
  # Không dùng max_corrupted_joint_frame_fraction chung cho 5 families
  spike_rate_cap: 0.06
  missing_sampling: exact_count
  missing_rate_cap: 0.18  # chỉ dùng nếu Bernoulli severe q=0.15
  max_gap_duration_seconds: 0.48
  min_visible_joint_fraction: 0.70  # report or rule, not hidden test filtering
  min_usable_frame_fraction: 0.90
  recovery_method: linear
  boundary_fill: nearest
  all_missing_joint_policy: reject_sample
  recenter_after_transform: false
```

Thông số không thuộc family hiện chạy được phép tồn tại trong shared YAML nhưng phải được log chính xác những field **đã thực sự dùng**. Ghi provenance code revision / original sample ID / seed / family / severity / units / L / anchor / sampled noise distribution / mask counts / frame count / segment repetition handling.

## 10.1 Training-time augmentation scheduler: single hoặc mixed corruption (feature-flagged)

**Mục tiêu:** Một recording có thể sạch, có 1 lỗi, hoặc có 2–3 lỗi khác nhau. Các lỗi mô phỏng camera/pose estimator phải **nhẹ, có kiểm soát**, không đánh mất motion signal. Giữ single-family mode để tái hiện v2 và các phép ablation; mixed mode là thí nghiệm mới, **không ghi đè policy cũ**.

```yaml
augmentation:
  enabled: false
  mode: mixed_bounded  # legacy_single | mixed_bounded
  train_only: true
  resample_each_epoch: true
  # Pilot; khóa bằng corrupted development validation trước final test
  n_families_probability: {0: 0.40, 1: 0.35, 2: 0.20, 3: 0.05}
  family_weights: {gaussian: 0.30, spike: 0.20, missing: 0.30, scale: 0.10, rotation: 0.10}
  severity_weights: {mild: 0.80, moderate: 0.20, severe: 0.00}
  sample_families_without_replacement: true
  # To avoid synthetic motion dominated by camera artifacts, at most one
  # geometric transform in a composite during pilot.
  max_geometric_families_per_sample: 1
  composite_budget:
    max_missing_fraction: 0.12
    max_spike_fraction: 0.025
    max_local_displacement_p95_body_scale: 0.05
    min_usable_frame_fraction: 0.90
    max_recovery_rejected_fraction: 0.02  # dataset-level audit, not hidden dropping
  on_budget_failure: no_op_and_log # not resample indefinitely
  clean_reference_immutable: true
```

**Giải thích xác suất:** 40% samples không bị phá; 35% một lỗi; 20% hai lỗi; 5% ba lỗi. Đây là **pilot** do nhóm đề xuất, không phải các tỷ lệ infant pose error đã được đo. `family_weights` là trọng số lấy mẫu **có điều kiện khi chọn một family**, lấy không hoàn lại; không có nghĩa 30% toàn bộ sample chắc chắn có Gaussian. Không mặc định kết hợp Gaussian + spike + missing + scale + rotation trong cùng sample.

**Budget phải áp dụng trên KẾT QUẢ KẾT HỢP**, không chỉ kiểm tra từng family riêng lẻ: tính tổng các site missing, spike, displacement phần local và mức độ còn quan sát được. Geometry (scale/rotation) có giới hạn biên độ riêng, không tính vào tỷ lệ joint-frame corrupted. Phân phối composite thực tế phải được thống kê; ghi cụ thể trường hợp fail/bỏ qua để tránh lệch phân phối âm thầm.

**Thứ tự gợi ý để tạo corruption:** geometry cố định cho segment (nếu được chọn) → Gaussian → spike → missing/gaps → recovery → optional preprocessing → recompute B → finite-check. Với benchmark có nhiều family, ghi danh sách family và thứ tự chính xác. Không tính `body_scale L` từ skeleton đã bị phá: luôn tính trên clean reference. Với Gaussian/spike dùng together, displacement audit đo trên skeleton quan sát được sau khi kết hợp. Khi missing, masked sites không được đưa vào displacement audit.

**Hai chế độ bắt buộc:**

1. `legacy_single`: policy v2 giữ nguyên 50% clean, 50% corrupted, chỉ 1 family với weights 0.30/0.20/0.30/0.10/0.10 và severity 0.70/0.30/0.00. Dùng để so sánh và bảo toàn backward compatibility.
2. `mixed_bounded`: dùng distribution mới. Chỉ bật bằng config explicit; thí nghiệm P3 riêng, **không đổi định nghĩa P0/P1/P2 cũ**.

Không corrupt labels; không dùng NaN cho model; severity `severe` chỉ nằm trong held-out stress-test. Chú ý augmentation chỉ diễn ra trên **train set**, không tạo thêm augmentation rows trong metadata hoặc thêm `augment_name` vào Feather.

### 10.1.1 Dataset integration: không phá mapping của `InfantMotionDataset`

Trong repo hiện tại (cần Codex kiểm tra trước), `modules/data.py` dùng metadata theo index và danh sách `augment_name`. Nếu nhân bản `self.samples` theo biến thể noise mà không nhân bản metadata đồng bộ, có thể gây lỗi mapping nhãn hoặc index out-of-range.

- **Một Feather file = một logical dataset sample**. Lấy label, subject ID và file metadata theo record/file key ổn định, **không** theo index trong danh sách augmentations.
- Load clean `J` → optional corruption on-the-fly / cached scenario → recovery/filter → tính `B` bằng đúng graph convention của model → trả `J+B` với shape sẵn có.
- **Pilot chỉ hỗ trợ JB-AAGCN**; chưa tính `V/A` trong model adapter. Để JBV/JBVA sang giai đoạn khác khi giải quyết valid-length và repetition seams.
- Không sửa `InfantMotionDataset` legacy nếu không cần. Ưu tiên `CorruptionDatasetWrapper` hoặc một adapter mới; **tắt cờ là trả đúng byte/tensor tương đương loader cũ**.
- Tạo regression test kiểm chứng ID ↔ corrected age với nhiều augment modes và test `num_workers` khác nhau.

### 10.1.2 Seed không phụ thuộc DataLoader scheduling

Train seed ổn định bằng SHA-256 từ `global_seed | fold | epoch | subject_id | sample_id | family | replicate` (cộng `draw_index` cho family selection). **Không dùng worker_id, process ID, Python hash() hoặc clock time.**

Dataset có `set_epoch(epoch)` và callback Lightning cập nhật trước epoch. Khi `persistent_workers=True`, phải kiểm chứng epoch state được chia sẻ đúng với worker processes (hoặc tái tạo DataLoader workers mỗi epoch); không mặc định setter ở main process tự cập nhật workers.

Unit test lặp lại cùng epoch, cùng sample với `num_workers=0,2,8`, kiểm tra corruption hashes như nhau. Với train split, đổi epoch có thể sinh noise khác; validation/test fixed scenario không phụ thuộc epoch.

## 10.2 Paired-severity deterministic corruption (mới)

Mục đích: cùng sample/replicate/family có **cùng latent randomness** ở mild/moderate/severe; thay severity chỉ đổi độ mạnh. Dùng stable SHA-256 → uint64 seed, **không dùng Python built-in `hash`**:

```text
experiment_seed | fold | sample_id | family | replicate
```

**Không đưa severity vào seed** cho evaluation paired-severity. Subject ID có thể thêm vào để tránh trùng sample_id; fold không được thay thế subject split. `attempt` chỉ xuất hiện ở trường hợp rejection hợp lệ và phải log.

- Gaussian: giữ cùng standard-normal tensor `Z`, nhân `alpha * L` theo severity.
- Spike: giữ cùng uniform site scores, direction vector và base magnitude quantile; tập điểm bị spike ở mild là subset của moderate/severe. Displacement range thay theo severity.
- Random missing: giữ cùng uniform site scores; chọn exact-count nhỏ nhất → mức mạnh hơn chứa toàn bộ missing của mức nhẹ.
- Temporal gap: giữ vị trí joint, anchor frame/center; mở rộng độ dài theo severity (không thay vị trí ngẫu nhiên). Kiểm tra nested mask nếu cùng số gaps.
- Scale: giữ cùng `u∈[-1,1]`, lấy `s=1+βu`; rotation giữ cùng signed `u`, `θ=θmax*u`.
- **Paired severity không đảm bảo downstream RMSE biến thiên đơn điệu**, chỉ loại bớt nhiễu do random realization.

## 10.3 Protocol v4 — split subject-level khóa trước training; inner development và outer test

**CRITICAL: repo gốc dùng `train_indices`/`val_indices` của 10-fold CV; cùng `val_loader` có thể vừa chọn checkpoint vừa báo cáo prediction. Do đó không được tự gọi `val_indices` là test độc lập trong protocol cũ.** Không sửa legacy reproduction. Tạo một **protocol nghiên cứu robustness mới** với outer-fold test và inner development split theo infant ID.

### 10.3.1 Freeze split và training protocol trước P0

Cho mỗi outer fold đang có:

```text
all subjects
 ├─ outer_test  = original outer val_indices (strictly untouched until final)
 └─ outer_train = original outer train_indices
     ├─ inner_train = fit model / on-the-fly augmentation
     └─ inner_dev   = checkpoint selection + severity/recovery/policy development
```

- **Trước khi train P0**, freeze: outer/inner subject ID membership, split seed, age-bin strategy, checkpoint metric `clean_inner_dev_MSE`, architecture, optimizer, batch size, learning-rate/schedule, epochs/training budget, aggregation convention và random seeds. P0 hoàn toàn có thể chạy song song với severity calibration **sau** khi freeze các mục này.
- Subject ID được resolve từ metadata/filename; **mọi segments của cùng infant cùng role**. Không reuse cùng recording hoặc corrupted replicas giữa các roles. `outer_test` không xuất hiện trong dataloader phục vụ tuning, pilot, visual samples hay inspect distribution-by-policy; chỉ truy cập tối thiểu metadata cần để xác nhận split không trùng và tạo manifest.
- **Inner split:** subject-level train/dev với age-aware approximate stratification. Tính 4–5 quantile bins trên tuổi thuộc `outer_train` ở cấp infant (không ở cấp segment), gộp bins nếu nhóm quá ít; stratify theo age bins, có thể thêm outcome 0/1 khi cỡ nhóm đủ. Không làm cho một stratum có quá ít subject dẫn tới lỗi split. Freeze và lưu cách xử lý ties/duplicates/không đồng nhất tuổi theo infant, số lượng subject và age ranges của mỗi role. Không đảm bảo exact age distribution vì sample size nhỏ.
- Lưu manifest bất biến `fold, subject_id, role, age, outcome, split_seed` và `split_manifest_sha256`, đảm bảo không duplicate subject across roles; giữ bản manifest riêng cho từng outer fold. Tuổi và outcome chỉ dùng stratification/descriptive audit; outcome 2/MNI không trộn vào training nếu baseline Holmberg loại khỏi age training.
- Mọi P0/P1/P2/P3 trong một outer fold phải dùng **cùng** inner/outer membership và cùng budget. Một baseline checkpoint đã chọn trên original outer val **không** là P0 đối chứng cho protocol mới; báo cáo nó riêng như replication.
- Đây là `outer CV + inner holdout`, **không phải full nested hyperparameter CV**; mô tả đúng khi viết báo cáo. Nếu nhiều outer folds được dùng để đánh giá, policy selection không được học từ các outer-test results của folds trước; khóa global policy trước khi mở bất kỳ outer test nào.

### 10.3.2 Phase A1 — development/calibration (chỉ inner_train + inner_dev)

```text
inner_dev clean J → fixed development scenarios (5 families × severities × replicates)
→ missing: fixed recovery → recompute B → development baseline checkpoint
→ pilot visualization + selection of severity and recovery/preprocessing candidates
→ lock corruption policy, selected pipelines, seeds, aggregation, clean-MAE guardrail
→ immutable config JSON/YAML + SHA-256 + source revision
```

- Có thể đánh giá baseline sensitivity trên **inner_dev**, nhưng tuyệt đối không trên `outer_test` trước config lock.
- Nếu cần calibrate severity trên data, dùng `inner_train`/`inner_dev` only. Trong fixed validation scenarios, các models có **cùng paired masks** và replicates.
- **Checkpoint selection (per run):** epoch có **thấp nhất clean inner-dev MSE** (`val_loss` theo code Holmberg); giữ nguyên cho P0/P1/P2/P3 và end-to-end preprocessing candidates. Không chuyển P1 sang MAE trong khi P0 dùng MSE. Tie-breaking: epoch thấp nhất.
- **Policy selection (separate from checkpoint):** compare frozen checkpoints trên corrupted inner-dev MAE hoặc `ΔMAE` theo pre-registered aggregate, với ràng buộc clean inner-dev MAE tăng không quá **5% tương đối** so với P0 (pilot engineering gate, không phải clinical threshold). Nếu P0 MAE gần 0, định nghĩa epsilon để tránh division by zero. Nếu không policy nào qua gate, giữ P0 và báo cáo thất bại. Không lựa chọn policy từ outer_test.
- Cần khóa severity/policy ở **cấp toàn study** trước khi final evaluation; hash config + split manifests + code version. Dù inner_dev được dùng để calibrate corruption, vẫn phải tránh lặp quá nhiều thử nghiệm rồi báo cáo hiệu quả như unbiased.

### 10.3.3 Phase B — training and two distinct preprocessing protocols

| Policy | Training | Vai trò |
|---|---|---|
| P0 | clean only | New clean-only nested-protocol baseline |
| P1 | 1 local family: gaussian/spike/missing | local robustness |
| P2 | 1 of five families | geometric contribution |
| P3 (optional) | mixed bounded, max 3 families | multi-failure robustness extension |

**Protocol R — Plug-in recovery với frozen baseline (đánh giá recovery at inference):**

```text
frozen P0 trained on clean original J+B
inner_dev/test corrupted J → fixed minimal recovery (linear/PCHIP etc.)
→ recompute B → frozen P0
```

- So sánh direct (nếu finite), linear recovery, PCHIP; boundary nearest fill; reject whole-joint-missing samples per frozen rule. **Không** gọi cách này là train-matched evidence cho toàn bộ pipeline 4554/3123. Chỉ kết luận `end_to_end_corruption_robustness`, `recovery_reconstruction_error`, `post_recovery_prediction_drift`; tránh claim AAGCN tự handle NaN.
- Nếu thêm smoothing hoặc normalization tại inference với frozen clean checkpoint, mô tả là **out-of-distribution sensitivity probe**, không phải một so sánh fair về lợi ích của preprocessing trên KA.

**Protocol E — End-to-end preprocessing, train/inference matched (đánh giá pipeline 4554/3123):**

```text
inner_train clean J → preprocessing candidate → compute B → train matched model
inner_dev clean/corrupted J → same candidate preprocessing → compute B
→ checkpoint using CLEAN inner-dev MSE; select policy on CORRUPTED inner-dev
outer_test clean/corrupted J → same candidate preprocessing → compute B
→ one-time final assessment only after lock
```

- Sử dụng same candidate on train and inference. Với missing, recovery xảy ra trước stages đòi đầu vào finite; xác định đúng operator order một lần. Retain unchanged P0 for baseline. Nếu preprocessing có fitted statistics, chỉ fit trên `inner_train`, không bao giờ trên dev/test.
- Không so trực tiếp model train-matched preprocessing với frozen baseline bị áp preprocessing chỉ khi inference rồi kết luận preprocessing ưu việt.
- Trong pilot **chỉ hỗ trợ `J+B`**, tuyệt đối không tính `V/A` trên 600-frame repetition nếu chưa có original `valid_length`/seams. J/B recomputation sau mọi modification của J. `augmentation.enabled=false` phải **delegate legacy loader trực tiếp** (parity); không tự read J/recompute B ở baseline disabled.

### 10.3.4 Phase A2/C — held-out outer-test, một lần sau lock

```text
Locked configs + same inner splits + selected P0/P1/P2/(P3) checkpoints
→ fixed frozen OUTER-TEST clean + corruption scenario_bank
→ report metrics by subject/segment/family/severity/replicate
```

- Bắt buộc report clean MAE/RMSE, corrupted MAE/RMSE, `ΔMAE`, prediction drift (paired clean vs corrupt), subject-level mean/median aggregation, recovery error và failure counts. Primary infant aggregation phải nhất quán với Holmberg; không coi segments là subjects độc lập.
- `prediction_drift = |y_pred_corrupt − y_pred_clean|` **không** thay thế age error against true label. Geometric transformation ≠ denoising error; giữ riêng geometric-invariance metrics và measurement-recovery metrics.
- Các metrics missing mô tả **pipeline có recovery + model**, không gán cho GNN khả năng xử lý NaN.
- Không chỉnh hyperparameters, severity, checkpoint, filter, threshold hoặc model selection sau khi xem test. Nếu phải đổi do bug, ghi rõ version và rerun **toàn bộ** policies một cách nhất quán; đánh dấu evaluation trước là invalid/exploratory.

## 10.4 Cache/reuse: preprocessing một lần, nhiều lượt train

**Đồng ý cache — nhưng cache đúng tầng.** Input là skeleton sequence, không phải raw RGB video. Các phép SG/Butterworth/imputation/centering được tính trên tọa độ `J`, sau đó `B` cần tái tính. Không nên lưu quá nhiều bản MP4 khi mục tiêu là train AAGCN: tốn dung lượng, mất masks/provenance và phải pose-estimate lại.

**Tầng cache:**

1. `raw_reference/` (read-only): clean Holmberg Feather, metadata. Không overwrite.
2. `scenario_bank/` (fixed evaluation): `J_corrupted`, masks, family, severity, replicate, seed, `L`, pivot, transform order. Dùng cho dev validation và final test **tách folder/splits**. Sinh trên split đã khóa; không đọc test trước Phase A2.
3. `recovery_cache/` (optional): `(scenario_id, recovery_config_hash)` → `J_recovered` và audit. Dùng nhiều lần khi so sánh các model, cùng recovery.
4. `preprocessing_cache/` (optional): `(scenario_id, recovery_hash, preprocess_config_hash, algorithm_version)` → `J_processed` và `B_recomputed`. Dùng để thử lại model/training với cấu hình bất biến.

**Không cache duy nhất một tập noise cố định cho toàn bộ robust training nếu mục tiêu là đa dạng augmentation qua epoch.** Chọn một trong hai:

- `on_the_fly`: generator deterministic theo epoch trong Dataset, cache clean/preprocessing pieces *trước augmentation* nếu hợp lệ.
- `epoch_bank`: tạo sẵn một tập finite scenarios **theo epoch**, ví dụ epoch 0–19, với seed khóa; có thể lưu disk và tái dùng giữa runs, nhưng tránh `same exact corrupted sample every epoch` làm giảm đa dạng. `epoch_bank` phải có clean samples tương ứng policy.

**Bài toán thứ tự:** cache `preprocess(clean J)` **không** tương đương `preprocess(corrupt J)`. Nếu nghiên cứu ability of a filter to repair corrupted skeleton, phải áp dụng filter **sau corruption** và tạo cache keyed theo corruption scenario. Cache ở bước nào đều phải lưu rõ thứ tự phép biến đổi.

**Cache correctness:** key có dataset version, subject/sample ID, split/fold, coordinate schema, joints/edges, FPS, corruption config hash, recovery config hash, preprocessing config hash và source-code revision; atomic writes, immutable artifacts, validation of shape/finite/checksum, cache invalidation khi đổi config; không đưa cached corruption vào `self.samples` để nhân bản nhãn. Cache theo split để tránh accidental leakage. Luôn có chế độ `cache.enabled=false` để kiểm tra parity.

**Cache phải đảm bảo kết quả tái lập**, không được sử dụng như lý do thay đổi phân phối augmentation hay âm thầm áp smoothing lên cả reference clean. Các bản được dùng làm đầu vào mô hình phải qua finite check. Nếu một sample bị reject, ghi log và loại/giữ theo policy đã khóa; không chọn ngầm sample dễ cho kết quả tốt.

## 11. Tests bắt buộc trước commit

1. **Seed determinism:** cùng input/config/subject/sample/seed → byte-identical output, khác seed → nhìn chung khác.
2. **Input immutability:** `X` gốc không bị sửa (including NaN positions), labels and metadata unchanged.
3. **Shape + dtype:** `(T,J,2)` và `(T,J,3)` đều hoạt động.
4. **Scale pivot fixed:** với pivot `c=(100,80)` và một joint `(140,80)`, `s=0.5` cho kết quả `(120,80)` **không phải `(70,40)`**; pivot đi qua phép biến đổi không đổi. Kiểm tra same scale across every frame, ratios/distances obey s.
5. **Rotation geometry:** quay 90° quanh pivot `(100,80)`: joint `(140,80)` → `(100,120)` theo trục Cartesian (nếu image y-down vẫn phải nhất quán convention được định nghĩa); `RᵀR=I`, distances preserved, same angle all frames.
6. **Missing:** only chosen sites NaN; masks correct; no zero padding; gaps lengths correct for FPS.
7. **Gaussian:** empirical std xấp xỉ αL trên lượng mẫu đủ lớn; no data-dependent drift injected.
8. **Spikes:** only chosen sites changed, average rate close to p with sample-size-appropriate tolerance.
9. **Not over-destruction:** default mild/moderate passes hard gates; severe logs/skips properly without silent parameter change.
10. **FOV:** strict mode never clips geometry; out-of-frame cases reject/resample.
11. **Holmberg JB adapter:** modify only `J`, recompute `B` after recovery/preprocessing; JBV/JBVA outside v1 pilot; no corrupted/clean stream mixing.
12. **No leakage:** fixed subject folds; noise is not shared across same child's train/test because child cannot cross split.

13. **No-NaN model path:** corrupt missing → linear/nearest recovery → recompute streams → `np.isfinite` trên toàn bộ input AAGCN; whole-joint-missing → reject và log.
14. **Holmberg neck pivot:** neck đã center vẫn gần 0 sau scale/rotation; midhip pivot trên Holmberg phải explicit non-default.
15. **Family-specific gates:** Gaussian dense không bị reject bởi 15% site cap; missing exact-count q=0.15 không bị rejection bias; global transforms exempt affected fraction.
16. **Paired severity:** Gaussian latent Z như nhau; missing/spike severe mask superset mild; scale/rotation cùng signed latent; stable SHA-256 seeds reproducible.
17. **Scheduler:** legacy_single backward compatible và mixed_bounded train-only; verify 0/1/2/3 family frequency, family without-replacement, composite budgets; no val/test augmentation; disabled mode byte-identical với baseline.
18. **Metrics:** mean và median infant aggregation không trộn segment; clean/corrupt drift đúng; separate `end_to_end_corruption_robustness` from `recovery_reconstruction_error`; all results traceable by seed/replicate.
19. **Split protocol:** outer test is original outer-val fold but NOT used for checkpoint/policy; freeze subject-level inner manifest BEFORE P0; no overlaps; age-bin stratification only on outer-train; config+split SHA-256 locked before A2; checkpoint lowest clean inner-dev MSE for all; policy chosen on corrupted inner-dev with clean-MAE guardrail.
20. **Dataset & workers:** 1 Feather = 1 logical sample, age keyed to sample ID; no augmentation rows in `self.samples`; deterministic outputs with `num_workers=0,2,8`, including `persistent_workers` or explicit workers-restart protocol.
21. **Cache:** cached vs no-cache tensors byte/close equivalent; config mutation invalidates cache; cache scenario IDs separate by split; `preprocess(corrupt(X))` cached in correct order; clean reference remains untouched.
22. **Mixed stress:** total corrupt budget checked after composition, sampled family sets unique, a clean subset remains; no hidden aggressive resampling.

## 12. Implementation order và acceptance criteria

1. Inspect repo: locate data loaders, constants/joint schema, preprocessing, model stream generator, config, tests. Print brief plan of files changed.
2. Commit 1: Implement `CorruptionConfig`, deterministic RNG, masks, `body_scale`, `pivot` and geometry tests.
3. Commit 2: Implement Gaussian and spike with unit tests.
4. Commit 3: Implement missing random/gap submodes with unit tests.
5. Commit 4: Implement scale + rotation, Holmberg neck-pivot and shape-preservation tests.
6. Commit 5: Add missing recovery wrapper + stream recomputation + no-NaN model-path tests.
7. Commit 6: Add paired severity RNG, family-specific gates, CLI/config, metadata + visuals.
8. Commit 7: Add feature-flagged train-only augmentation scheduler + fixed evaluation protocol and metric scripts.
9. Add two protocol tests: (i) frozen-P0 plug-in recovery never feeds NaN to model; (ii) train-matched preprocessing applies the same configured operators to train/dev/test and recalculates B. Assert outer-test data are never loaded during development.
10. Run test suite in existing environment; produce a runbook. Do not train full 10-fold sweep unless explicitly requested. **Do not start large-scale training unless explicitly asked.**

**Definition of done:** frozen outer/inner subject-level split manifests before P0, consistent checkpoint-selection by clean inner-dev MSE, explicit Protocol R (frozen model recovery) vs Protocol E (train-matched preprocessing); 5 families implemented; masks/metadata deterministic; Holmberg neck pivot; family-specific gates; NaN-free recovered AAGCN input; J/B recomputation only in v1; paired severity; legacy and optional mixed train-only augmentation scheduler feature-flagged; validation-first and locked-test protocol; deterministic scenario/cache keys; baseline unaffected when disabled; pilot corruption statistics and visualizations. No invented clinical validity claims.

## 13. Tài liệu cơ sở và phạm vi suy luận

- **Holmberg et al. (2025),** *Learning Developmental Age From 3D Infant Kinetics Using Adaptive Graph Neural Networks*. Paper/preprocessing: https://arxiv.org/abs/2402.14400 . Cho biết preprocessing centering/rotation và biến thể mô hình; **không** chứng minh các severity pilot ở đây là chuẩn infant.
- **Kaur et al. (2026),** *Machine learning approaches in automated infant General Movements Assessment: A scoping review*. https://doi.org/10.1111/dmcn.70308 . Nêu ảnh hưởng của camera distance, body size và lỗi landmark, cũng như nguy cơ filtering làm méo chuyển động. Cơ sở infant-domain để chọn failure modes, **không phải calibration của noise distribution**.
- **Mathe et al. (2025),** *Leveraging Artificial Occluded Samples for Data Augmentation in Human Activity Recognition*. https://doi.org/10.3390/s25041163 . Cơ sở về structured body-part removal nhưng là human action recognition.
- **Wu et al. (2025),** *A robust two-stage framework for human skeleton action recognition with GAIN and masked autoencoder*. https://doi.org/10.1016/j.neucom.2025.129433 . Cơ sở về missing keypoints và frame-joint motion masking; không phải infant-specific.
- **Rubin (1976),** *Inference and Missing Data*. https://doi.org/10.1093/biomet/63.3.581 . Nền tảng lý thuyết missing-data mechanisms; simulation MCAR ở đây là lựa chọn thí nghiệm.
- **Pinhole camera model:** trong trường hợp hình chiếu xấp xỉ `h_image ≈ f H_real/Z`, global 2D scaling có thể mô phỏng thay đổi apparent size; không mô phỏng được perspective, occlusion hoặc thay đổi depth từng joint.

**Phân biệt chứng cứ:** mọi severity, thresholds, tolerance gates và lựa chọn pivot trong tài liệu này là **đề xuất thiết kế** cần pilot calibration; không phải giá trị chuẩn được các nguồn trên xác lập cho infant.
