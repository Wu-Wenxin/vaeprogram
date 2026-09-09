"""実験設定の dataclass スキーマ.

YAML から読み込み，``ExperimentConfig`` を起点に各サブ config に分岐する．
ネストは2段までに留め，型は標準ライブラリのプリミティブと ``Literal`` のみ．
这个项目允许配置哪些东西？
每个配置项叫什么？
默认值是什么？
哪些是必须写的？
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataConfig:
    """合成データ生成の設定.

    Attributes:
        name: データ生成器の名前．``synthetic_emg`` / ``synthetic_lever`` /
            ``synthetic_spec`` / ``synthetic_pose`` のいずれか．
        channels: チャネル数（1D系で C, 姿勢系で D）．
        length: 系列長 T．
        n_classes: クラス数（None なら 4 クラスをデフォルトとして適用）．
        batch_size: 学習バッチサイズ．
        val_ratio: 検証セットの割合．
        seed: 生成器のシード．
        stride: 実 EMG の sliding window stride（``real_emg`` のみ参照）．
    """

    name: str
    channels: int
    length: int
    n_classes: int
    data_dir: str
    stride: int

    batch_size: int = 64  # 每个训练批次的窗口数量。
    val_ratio: float = 0.2  # 验证集比例。
    split_by: Literal[
        "sample",
        "file",
        "file_stratified",
    ] = "file" # 数据划分方式
    seed: int = 0  # 控制训练集和验证集的随机划分


@dataclass
class ModelConfig:
    """モデル種別と backbone の設定.
    VAE模型相关配置。
    Attributes:
        kind: VAE の種類．
        backbone: encoder/decoder の backbone．
        latent_dim: 潜在空間の次元数 D．
        hidden_channels: encoder の各段のチャネル数（decoder はミラー）．
        cond_dim: Conditional VAE の条件次元（kind=conditional/classified で使用）．
        cond_inject: 条件の注入位置（``encoder`` / ``decoder`` / ``both``）．
        n_components: GMM-VAE の混合数．
        sequence: 系列VAE 用の追加設定．
    """

    kind: Literal["vanilla"] = "vanilla"
    backbone: Literal["simple1d", "resnet1d"] = "simple1d"
    latent_dim: int = 8
    # 每个卷积块的输出通道数。
    hidden_channels: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128]
    )


@dataclass
class ReconConfig:
    """再構成損失の設定."""
    # 重建损失的计算方法。
    kind: Literal["mse", "l1", "gauss_nll"] = "mse"
    # 重建损失在总损失中的权重。
    weight: float = 1.0
    

@dataclass
class AnnealConfig:
    """KL 重みのアニーリング設定.
    KL损失权重变化方式。
    ``kind`` が ``constant`` のときは ``weight`` をそのまま使います．
    ``linear`` は 0 → ``weight`` まで ``end_step`` ステップで線形に上げます．
    ``cyclic`` は 0 → ``weight`` を ``cycle_steps`` 周期で繰り返します．
    """
    # 权重变化方式：固定、线性增加或周期变化。
    kind: Literal["constant", "linear", "cyclic"] = "constant"
    end_step: int = 1000  # linear模式下，经过多少步增加到目标权重。


@dataclass
class KLConfig:
    """KL 損失の設定."""

    weight: float = 1.0 # KL损失在总损失中的目标权重。
    free_bits: float = 0.0 # 每个潜在维度允许忽略的最小KL值。
    warmup_steps: int = 0 # 训练开始后，多少步内不加入KL损失。
    anneal: AnnealConfig = field(default_factory=AnnealConfig) # KL权重的变化方式。

@dataclass
class ClassifyConfig:
    """分類ヘッド用の補助損失（CrossEntropy）．
    分类辅助损失配置。
    クラス数は ``DataConfig.n_classes`` 側で一元管理し，ここでは重みのみ持つ．
    """

    weight: float = 1.0


@dataclass
class SpectralConfig:
    """周波数損失の設定.
    频域重建损失配置。"""

    weight: float = 0.1
    n_fft: int = 64
    hop_length: int = 16


@dataclass
class StructureConfig:
    """潜在の構造化損失.
    潜在空间结构损失配置。"""

    kind: Literal["covariance", "orthogonal", "tc"] = "covariance"
    weight: float = 0.1


@dataclass
class TemporalSmoothnessConfig:
    """窓系列潜在の時間方向ペナルティ（m2m SequenceVAE 専用）.
    潜在变量时间平滑损失配置。
    ``mu`` / ``z`` が ``(B, W, D)`` のときのみ作用する．それ以外（m2o の
    ``(B, D)``）では builder で 0 を返してスキップされる．

    Attributes:
        weight: ペナルティの重み．0 で無効．
        order: 1 で連続性ペナルティ ``||μ_w - μ_{w-1}||²``，
            2 で加速度ペナルティ ``||μ_{w+1} - 2μ_w + μ_{w-1}||²``．
        target: ``mu`` 推奨．``z`` は再パラメータ化ノイズ込みで学習信号が
            歪むが，ablation 用に選択可能．
    """

    weight: float = 0.0
    order: int = 1
    target: Literal["mu", "z"] = "mu"


@dataclass
class IdleAnchorConfig:
    """idle anchor lossの設定。"""

    weight: float = 0.0
    # 几何损失使用后验分布中心 mu，或使用重参数化后的采样 z。
    input: Literal["mu", "z"] = "mu"


@dataclass
class AngularPrototypeConfig:
    """angular prototype lossの設定。"""

    weight: float = 0.0
    temperature: float = 0.1
    # 默认使用稳定的 mu；z 用于带采样噪声的对照实验。
    input: Literal["mu", "z"] = "mu"


@dataclass
class DistanceActivityConfig:
    """idleからの距離とEMG活動量を対応させる損失の設定。"""

    weight: float = 0.0
    # 先生の案ではサンプリング済み潜在変数zを使用する。
    input: Literal["mu", "z"] = "z"


@dataclass
class LossConfig:
    """損失全体の構成.
    所有损失配置的集合。"""

    # VAE必须使用的重建损失。
    recon: ReconConfig = field(default_factory=ReconConfig)
    # VAE必须使用的KL损失。
    kl: KLConfig = field(default_factory=KLConfig)
    idle_anchor: IdleAnchorConfig = field(default_factory=IdleAnchorConfig)
    angular_prototype: AngularPrototypeConfig = field(
        default_factory=AngularPrototypeConfig
    )
    distance_activity: DistanceActivityConfig = field(
        default_factory=DistanceActivityConfig
    )


@dataclass
class OptimConfig:
    """最適化の設定.
    优化器配置。"""
    
    lr: float = 1e-3 # Adam优化器的学习率。
    weight_decay: float = 0.0 # 权重衰减强度。
    betas: tuple[float, float] = (0.9, 0.999) # Adam优化器的一阶和二阶动量参数。


@dataclass
class TrainConfig:
    """学習ループの設定.
    训练过程配置。"""

    epochs: int = 20
    grad_clip: float | None = 1.0 # 梯度裁剪上限，None表示不裁剪。
    output_dir: str = "outputs/run"
    seed: int = 0
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

@dataclass
class VizConfig:
    """可视化相关配置。"""

    # latent图使用验证集或全部数据。
    latent_source: Literal["val", "all"] = "val"

@dataclass
class ExperimentConfig:
    """実験全体の設定（YAML のルート）.
    一次完整实验的全部配置。"""
    
    data: DataConfig  # 真实EMG数据配置，必须在YAML中提供。
    name: str = "experiment"
    title: str | None = None # added
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    viz: VizConfig = field(default_factory=VizConfig)
