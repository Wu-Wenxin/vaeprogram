# vae-wwx

针对真实肌电（EMG）信号的极简 VAE 实验框架。目标是学习可解释的潜在空间，使手势类别在低维隐空间中自然分离。

---

## 项目结构

```
vae-wwx/
├── configs/              YAML 配置文件
│   └── real_emg.yaml     真实 EMG 实验的默认配置
├── data/                 数据集（不纳入版本管理）
│   ├── real-emg/
│   └── rectify_smooth/   预处理后的数据（整流+平滑）
├── examples/             示例运行脚本
│   ├── 01_test_model.py  模型基础测试
│   ├── 02_run_from_yaml.py  完整训练流程（推荐入口）
│   └── 03_compare_stride.py  步幅消融对照实验
├── src/vae_wwx/          核心源码
│   ├── config/           配置加载与结构验证
│   ├── data/             数据集读取与预处理
│   ├── losses/           组合损失函数
│   ├── models/           VAE 模型结构
│   ├── train/            训练循环、Trainer、Checkpoint
│   ├── utils/            工具函数（随机种子、日志等）
│   └── viz/              可视化（潜在空间、重建、损失曲线）
├── tests/                单元测试
├── outputs/              训练输出（不纳入版本管理）
└── pyproject.toml        项目配置（依赖、构建、lint）
```

---

## 数据流

```
YAML 配置文件
    ↓ config/loader.py 读取
普通 dict
    ↓ config/schema.py 验证格式
ExperimentConfig 对象
    ↓
data/  →  model/  →  losses/  →  train/  →  viz/
```

**模型内部前向传播：**

```
x  (B, C, L)
  ↓ encoder（1D 卷积 backbone）
feature
  ↓ latent_head
μ, log σ²
  ↓ reparameterize（采样）
z  (B, D)
  ↓ decoder
x̂  (B, C, L)
```

---

## 环境安装

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在项目根目录创建虚拟环境并安装依赖
uv sync
```

**主要依赖：** Python ≥ 3.11、PyTorch ≥ 2.2、NumPy、SciPy、PyYAML、Matplotlib

---

## 快速开始

### 训练实验

```bash
# 使用默认配置训练
uv run python examples/02_run_from_yaml.py --config configs/real_emg.yaml

# 临时覆盖配置参数（dot notation）
uv run python examples/02_run_from_yaml.py --config configs/real_emg.yaml \
    --override model.latent_dim=4 train.epochs=100 loss.kl.weight=0.1

# 从 checkpoint 断点续训
uv run python examples/02_run_from_yaml.py --config configs/real_emg.yaml \
    --resume outputs/.../model.ckpt
```

训练完成后，输出目录（`train.output_dir`）中包含：
- `recon(...).png` — 重建效果示例图
- `latent_(...).png` — 潜在空间散点图（按类别着色）
- `curves(...).png` — 训练/验证损失曲线

### 步幅消融实验

```bash
# 在相同训练步数下对比 stride=250 与 stride=500 的效果
uv run python examples/03_compare_stride.py --config configs/real_emg.yaml
```

输出保存在 `outputs/compare_stride/` 下，含对比图与 `comparison.md` 摘要。

---

## YAML 配置说明

以 `configs/real_emg.yaml` 为例：

```yaml
data:
  data_dir: data/rectify_smooth/exp0703_hirota_trial1
  channels: 8          # EMG 通道数
  length: 500          # 窗口长度（采样点）
  stride: 250          # 滑动步幅
  n_classes: 5         # 类别数（0=idle，1-4=手势动作）
  batch_size: 64
  val_ratio: 0.2
  split_by: file       # 按文件划分训练/验证集

model:
  kind: vanilla        # 目前仅支持 vanilla VAE
  backbone: simple1d   # 1D 卷积 backbone
  latent_dim: 2        # 潜在空间维度
  hidden_channels: [16, 32, 64, 128]

loss:
  recon:
    kind: mse
    weight: 1.0
  kl:
    weight: 0.05       # β-VAE 权重
    warmup_steps: 100
    anneal:
      kind: linear
      end_step: 500
  idle_anchor:         # 将 idle 类拉向原点
    weight: 0.0
  angular_prototype:   # 促进非 idle 类的方向分离
    weight: 0.0
    temperature: 0.1
  distance_activity:   # 最大化潜在距离与 EMG 振幅的皮尔逊相关
    weight: 0.0

train:
  epochs: 50
  lr: 0.005
  grad_clip: 1.0
  output_dir: outputs/my_run
  seed: 0
  device: auto         # auto / cpu / cuda / mps
```

---

## 损失函数

| 损失项 | 说明 |
|---|---|
| `recon` (MSE) | 重建损失，主要训练信号 |
| `kl` | KL 散度，配合 β 权重与线性退火 |
| `idle_anchor` | 将 idle 类（class 0）的潜在表征拉向原点 |
| `angular_prototype` | 为各动作类分配可学习的方向原型，通过余弦相似度促进方向分离，**不影响潜在距离** |
| `distance_activity` | 最大化潜在距离 $r_t$ 与 EMG RMS 振幅的皮尔逊相关，不依赖标签 |

各损失项权重设为 0 时自动跳过，不影响训练速度。

---

## 数据格式

- 原始 EMG 数据经整流+平滑预处理后保存为 `*_labeled.csv`
- 每行格式：`[sample_index, ch1, ch2, ..., ch8, label]`
- 标签：0 = idle（静止），1–4 = 手势动作类别
- 数据集按文件（recording trial）划分训练/验证集，避免时序泄露

---

## 测试

```bash
uv run pytest
```

主要测试：数据集标准化一致性（`tests/test_emg_standardization.py`）——验证训练/验证集标准化参数仅由训练集计算，不包含验证集信息。

---

## 开发

```bash
# 代码格式检查与修复
uv run ruff check src/ examples/ tests/
uv run ruff format src/ examples/ tests/
```

---

## 输出目录结构（单次训练）

```
outputs/<run_name>/
├── model.ckpt              最终模型权重
├── recon(...).png          重建示例图
├── latent_val(...).png     验证集潜在空间散点图
├── latent_all(...).png     全量数据潜在空间散点图（可选）
└── curves(...).png         损失曲线
```


## 完整预处理流程

### 第一步：整流平滑

```python
uv run python src/vae_wwx/data/preprocess_real_emg.py data/real-emg/<file>
```
输出 _filtered.csv 保存到 data/rectify_smooth/<同名文件夹>/

### 第二步：自动标注动作段

```python
uv run python src/vae_wwx/data/label_segments.py data/rectify_smooth/<file>
```
输出 _labeled.csv、_segments.csv、诊断图

### 第三步（如果需要）：手动修正失败的标注
自动标注失败时标签写 -1，看 pic/ 里的诊断图，手动改 _segments.csv，再跑：

```python
uv run python src/vae_wwx/data/apply_segments.py data/rectify_smooth/<file>
```

运行代码
```python
uv run python examples/02_run_from_yaml.py --config configs/real_emg.yaml
```