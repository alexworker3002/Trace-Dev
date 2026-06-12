# TRACE-CT Agent Update Guide

本文件用于指导工程 Agent 将 TRACE-CT 文档中的最新架构修改同步到代码、配置、训练状态机、审计日志和实验验证流程中。核心更新目标是：在保持原有安全审计、残差门控和动态目标机制的基础上，新增 **D 模块去噪强度放行机制**，防止模型退化为对原始中心切片的轻微修正。

---

## 1. 更新目标

当前框架已经较好地防御了以下风险：

- 邻层高频泄漏；
- 残差结构污染；
- pseudo target 抹除小结构；
- 双反馈闭环振荡；
- 训练--推理主体不一致。

但新增风险是：

> D 模块可能过于保守，最终满足结构安全，却几乎不去除原始中心层噪声，即 `D(x_h, c_h, p_h) ≈ x_h`。

本次更新必须加入：

1. `G4.5 Denoising Strength Audit` 阶段；
2. `DenoisingStrengthController`；
3. D-as-noise-estimator 输出形式；
4. 欠去噪/过平滑/结构删除三类失败判定；
5. `denoising_strength_audit.json` 日志；
6. 训练状态机对 G5/G6/G7 的硬放行控制。

---

## 2. 新增核心概念

### 2.1 Conservative Identity Collapse

定义：

```text
s_hat = D_theta(x_h, c_h, p_h) ≈ x_h
```

表现：

- homogeneous ROI 噪声标准差几乎不下降；
- NPS amplitude 几乎不变；
- output residual `x_h - s_hat` 能量接近 0；
- 视觉去噪不明显；
- 结构保持率很高，但只是因为模型没动；
- D/P agreement 高，但二者都靠近 noisy REG。

### 2.2 D-as-noise-estimator

推荐将 D 模块从直接输出 clean image 改为显式噪声估计：

```text
n_hat = N_theta(x_h, c_h, p_h)
s_hat = x_h - G_den * n_hat
```

其中：

- `n_hat`: 模型估计应被移除的噪声；
- `G_den`: denoising gate / strength map；
- `s_hat`: 最终输出；
- `x_h - s_hat`: removed residual，必须被审计其是否像噪声。

禁止将 `context_gate` 和 `denoise_gate` 合并为同一个 gate。

---

## 3. 代码结构更新建议

建议新增或修改以下模块。

```text
trace_ct/
  models/
    refiner.py                  # 修改 D 为 noise-estimator form
    denoising_strength.py        # 新增 DenoisingStrengthController

  audit/
    denoising_strength_audit.py  # 新增 G4.5 审计指标
    residual_audit.py            # 保持原有 residual audit
    proposal_audit.py            # 保持 P qualification

  train/
    train_g4_residual.py         # 保持 residual-gated training
    audit_g45_strength.py        # 新增独立 G4.5 阶段
    train_g5_proposal.py         # 只有 G4.5 通过后允许执行
    train_g6_target_refine.py    # 只有 G5 通过后允许执行
    train_g7_alternating.py      # 每个 cycle 后重新跑 strength audit

  eval/
    metrics_noise.py             # 增加 NPS amplitude/shape metrics
    metrics_structure.py         # 增加 residual-edge correlation
    synthetic_lesion.py          # 复用/扩展 lesion retention

  configs/
    stage_g45_strength.yaml      # 新增配置
```

---

## 4. 必须新增的指标

### 4.1 Homogeneous Denoising Ratio

```text
r_D_hom = Std(Pi_N(s_hat) over Omega_hom) / Std(Pi_N(x_h) over Omega_hom)
```

默认判定：

```text
0.65 <= r_D_hom <= 0.85       # 合理去噪区间
r_D_hom > 0.90                # 欠去噪 / identity-like
r_D_hom < 0.60                # 可能过平滑
```

### 4.2 Output Residual Energy Ratio

```text
e_D = ||Pi_N(x_h - s_hat)||_2 / ||Pi_N(x_h)||_2
```

默认判定：

```text
0.10 <= e_D <= 0.35           # 合理输出残差能量
e_D < 0.05                    # 几乎没有去噪
```

### 4.3 Residual-Edge Correlation

```text
eta_res_edge = abs(Corr(grad(x_h - s_hat), grad(x_h)))
```

默认判定：

```text
eta_res_edge <= 0.10          # 通过
eta_res_edge > 0.10           # 可能删除结构
```

### 4.4 NPS Amplitude Reduction

```text
A_NPS = Integral_NPS(s_hat, noise_band) / Integral_NPS(x_h, noise_band)
```

默认判定：

```text
0.50 <= A_NPS <= 0.85
```

### 4.5 NPS Shape Distance

```text
d_shape = L1(normalized_NPS(s_hat) - normalized_NPS(x_h))
```

默认判定：

```text
d_shape <= 0.15
```

### 4.6 Synthetic Lesion Retention

```text
c_lesion = contrast(s_hat, lesion_roi) / contrast(x_h, lesion_roi)
```

默认判定：

```text
c_lesion >= 0.90
```

---

## 5. 新增 G4.5 阶段

### 5.1 阶段位置

G4.5 位于 G4 和 G5 之间：

```text
G0  data audit
G1  masked single-slice baseline
G2  low-pass context validation
G3  residual pool offline audit
G4  audited residual-gated training
G4.5 denoising strength audit      # 新增
G5  proposal qualification
G6  local dynamic target refinement
G7  short-cycle alternation
G8  full limited release
```

### 5.2 G4.5 禁止事项

在 G4.5 中禁止：

- 启用 dynamic target；
- 用 P proposal 替换 target；
- 联合训练 P 和 D；
- 用 HR 作为训练监督；
- 只看 validation loss 放行。

### 5.3 G4.5 输入

```text
checkpoint: D_theta after G4
input: x_h, c_h, optional p_h only as diagnostic condition
masks: Omega_hom, structure_risk_mask, lesion_test_mask
```

### 5.4 G4.5 输出

必须生成：

```text
denoising_strength_audit.json
```

字段见第 8 节。

### 5.5 G4.5 放行条件

必须同时满足：

```text
0.65 <= r_D_hom <= 0.85
e_D >= 0.10
eta_res_edge <= 0.10
c_lesion >= 0.90
A_NPS in [0.50, 0.85]
d_shape <= 0.15
```

如果失败，禁止进入 G5/G6/G7。

---

## 6. 失败分类与修复动作

### 6.1 欠去噪 / Conservative Identity Collapse

触发条件：

```text
r_D_hom > 0.90 or e_D < 0.05
```

修复动作：

1. 增加 homogeneous denoising loss；
2. 为 `G_den` 设置低风险区域下界；
3. 提高 `W_hom` 区域训练权重；
4. 适度扩大 residual injection strength，但仍受 `S_t` 控制；
5. 检查 residual training 是否只学会去 extra noise；
6. 降低过强的 structure anchor，但只限非风险区域；
7. 不允许继续长训替代修复。

### 6.2 过平滑

触发条件：

```text
r_D_hom < 0.60 or c_lesion < 0.90
```

修复动作：

1. 降低 homogeneous denoising loss；
2. 缩小 `W_hom`；
3. 增强 edge/lesion/structure anchor；
4. 提高 structure-risk mask 敏感度；
5. 降低 `G_den` 上限。

### 6.3 结构删除

触发条件：

```text
eta_res_edge > 0.10
```

修复动作：

1. 降低结构风险区域的 denoise gate；
2. 强化 residual-edge penalty；
3. 检查 `x_h - s_hat` 是否包含边缘；
4. 在 structure-risk mask 内强制 `G_den -> 0` 或极小值；
5. 重新执行 synthetic lesion test。

---

## 7. 训练损失更新

### 7.1 Homogeneous Active Denoising Loss

新增：

```text
L_hom_den = ||W_hom * Pi_N(s_hat)||_1
          + lambda_anchor * ||W_hom * Pi_lo(s_hat - x_h)||_1
```

用途：

- 第一项提供主动去噪压力；
- 第二项防止低频 HU 漂移。

### 7.2 Strength Interval Loss

新增：

```text
L_strength = relu(r_D_hom - tau_max)^2
           + relu(tau_min - r_D_hom)^2
```

默认：

```text
tau_min = 0.65
tau_max = 0.85
```

### 7.3 Under-denoising Single-sided Loss

可选 warmup：

```text
L_under = relu(r_D_hom - 0.90)^2
```

用途：只先防止 D 靠近 identity。

### 7.4 总损失接入原则

不要在全图启用强度损失。只允许在：

```text
W_hom == 1 and structure_risk_mask == 0
```

或低风险区域启用。

---

## 8. `denoising_strength_audit.json` schema

建议字段：

```json
{
  "stage": "G4.5",
  "checkpoint": "path_or_hash",
  "dataset_split": "validation_or_audit",
  "num_volumes": 0,
  "num_patches": 0,
  "metrics": {
    "r_D_hom_mean": 0.0,
    "r_D_hom_p05": 0.0,
    "r_D_hom_p50": 0.0,
    "r_D_hom_p95": 0.0,
    "e_D_mean": 0.0,
    "A_NPS_mean": 0.0,
    "d_shape_mean": 0.0,
    "eta_res_edge_mean": 0.0,
    "c_lesion_mean": 0.0
  },
  "flags": {
    "identity_collapse": false,
    "over_smoothing": false,
    "structure_deletion": false,
    "release_D": false
  },
  "thresholds": {
    "r_D_hom_min": 0.65,
    "r_D_hom_max": 0.85,
    "r_D_hom_identity": 0.90,
    "e_D_min": 0.10,
    "eta_res_edge_max": 0.10,
    "c_lesion_min": 0.90,
    "d_shape_max": 0.15
  },
  "recommended_action": "hold_or_release_or_rollback"
}
```

---

## 9. 状态机更新

训练状态机必须加入硬门控：

```python
if stage in ["G5", "G6", "G7"]:
    assert denoising_strength_audit["flags"]["release_D"] is True
```

G7 每个 cycle 后必须重新检查：

```python
run_denoising_strength_audit()
if identity_collapse or over_smoothing or structure_deletion:
    rollback_to_previous_checkpoint()
    freeze_or_recalibrate_modules()
```

---

## 10. 配置文件新增项

新增 `configs/stage_g45_strength.yaml`：

```yaml
stage: G4.5
input:
  checkpoint: null
  use_dynamic_target: false
  use_proposal_as_target: false
  use_hr_as_training_label: false

metrics:
  compute_r_D_hom: true
  compute_e_D: true
  compute_nps_amplitude: true
  compute_nps_shape: true
  compute_residual_edge_corr: true
  compute_lesion_retention: true

thresholds:
  r_D_hom_min: 0.65
  r_D_hom_max: 0.85
  r_D_hom_identity: 0.90
  e_D_min: 0.10
  eta_res_edge_max: 0.10
  c_lesion_min: 0.90
  d_shape_max: 0.15

fallback:
  block_next_stages_on_fail: true
  rollback_on_cycle_fail: true
  allow_continue_training_without_release: false
```

---

## 11. Agent 执行顺序

按照以下顺序更新，不要跳步：

1. 修改 `models/refiner.py`，让 D 显式输出 `noise_estimate`, `denoise_gate`, `removed_residual`, `s_hat`。
2. 新增 `models/denoising_strength.py`，实现 `DenoisingStrengthController`。
3. 新增 `audit/denoising_strength_audit.py`，实现 G4.5 指标。
4. 新增 `train/audit_g45_strength.py`，作为独立 CLI/stage。
5. 修改训练状态机：G4 后必须进入 G4.5；G4.5 未通过不得进入 G5/G6/G7。
6. 修改 G7 cycle：每个 cycle 后重跑 G4.5 audit。
7. 新增 `configs/stage_g45_strength.yaml`。
8. 修改报告生成逻辑，加入 strength report panel。
9. 更新单元测试和小规模 smoke test。
10. 更新 README 或实验协议说明。

---

## 12. 最小测试清单

必须通过以下测试：

```text
[ ] D output includes noise_estimate, denoise_gate, removed_residual, s_hat
[ ] denoising_strength_audit.py can run without P proposal
[ ] G4.5 fails when D is identity mapping
[ ] G4.5 fails when D over-smooths synthetic lesion
[ ] G4.5 fails when removed residual correlates with edges
[ ] G4.5 passes on a controlled synthetic denoising case
[ ] G5/G6/G7 are blocked if release_D is false
[ ] G7 reruns strength audit after every cycle
[ ] denoising_strength_audit.json is saved and contains required fields
[ ] No HR label is used as training target in G4.5
```

---

## 13. 禁止行为

Agent 不得执行以下操作：

- 不得一开始联合训练所有模块；
- 不得跳过 G4.5 直接进入 proposal 或 dynamic target；
- 不得用 HR 作为 G4.5 的训练监督；
- 不得只用 validation loss 判定 D 是否可放行；
- 不得把 context gate 当作 denoising gate；
- 不得在 structure-risk 区域强制去噪；
- 不得用继续长训替代 identity collapse 修复；
- 不得在 G4.5 失败时继续 G7 交替训练。

---

## 14. 验收标准

本次更新完成后，工程仓库应满足：

1. 文档中的 G4.5 阶段在代码状态机中真实存在；
2. D 模块输出可审计的 removed residual；
3. 欠去噪、过平滑、结构删除三类失败能被自动标记；
4. `denoising_strength_audit.json` 可复现生成；
5. G5/G6/G7 受到 `release_D` 硬门控；
6. 实验报告中同时展示：input、output、removed residual、denoise gate、hom mask、structure-risk mask；
7. short-run 即可发现 D 是否只是中心切片微调。

---

## 15. 对论文/理论文档的对应变动

本次代码更新对应理论文档中的以下新增内容：

- `Denoising Strength Controller`；
- `G4.5 D 模块去噪强度审计`；
- `Conservative Identity Collapse` 失败模式；
- `D-as-noise-estimator` 输出形式；
- `r_D^{hom}`, `e_D`, `A_NPS`, `d_shape`, `eta_res-edge`, `c_lesion` 指标；
- G5/G6/G7 的硬放行条件；
- `denoising_strength_audit.json` 证据日志。

