from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IdleAnchorLoss(nn.Module):
    """idleクラスの潜在変数を原点付近に固定する。
    把 idle（class 0）的潜在均值拉向原点。"""

    def __init__(self, idle_label: int = 0) -> None:
        super().__init__()
        self.idle_label = idle_label

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        # 找出 class 0
        idle_mask = labels == self.idle_label

        # 当前 batch 没有 class 0 时，返回 0
        if idle_mask.sum() == 0:
            return z.sum() * 0.0

        # 计算 class 0 的 z 到原点的平方距离
        # 把每个样本的所有潜在维度加起来
        squared_distance = z[idle_mask].pow(2).sum(dim=-1)

        # 对所有 class 0 样本求平均
        return squared_distance.mean()


class AngularPrototypeLoss(nn.Module):
    """各クラスに1方向を割り当て、方向の分離を促す。距離には影響しない。
    让 class 1～4 分别靠近各自的方向 Prototype。
    zを単位ベクトルに正規化してからコサイン類似度を計算するため、
    潜在空間上の「方向」のみを励まし、「距離」には一切影響しない。

    prototypesは学習可能なパラメータとして保持されるため、
    optimizerのパラメータリストに含める必要がある点に注意。

    Args:
        latent_dim: 潜在空間の次元数
        num_classes: 動作クラス数（idle以外）
        idle_label: idleクラスのラベル値
        temperature: softmaxの温度（小さいほど強い分離を要求）
        mode: prototype 的决定方式
            "learned"   : 老师示例的做法（默认）。prototype 是可学习参数，
                          用 torch.randn 随机初始化，由优化器更新。
            "class_mean": prototype = 各动作当前的平均方向。
                          不接收梯度，只由数据决定，排列方式完全交给数据。
        momentum: class_mean 时旧累计的保留比例（默认 0.9）
    """
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        idle_label: int = 0,
        temperature: float = 0.1,
        mode: str = "learned",
        momentum: float = 0.9,
    ) -> None:
        super().__init__()

        self.idle_label = idle_label
        self.temperature = temperature
        self.mode = mode
        self.momentum = momentum

        if mode == "learned":
            # 学習可能なプロトタイプ方向。
            # forward内で単位ベクトルに正規化する。
            self.prototypes = nn.Parameter(
                torch.randn(num_classes, latent_dim)
            )
        elif mode == "class_mean":
            # proto_sum：每个动作“单位方向向量”的累计和（旧的部分每次乘 momentum 淡化）。
            # 用 register_buffer 而不是 nn.Parameter：
            #   - 不会被优化器更新（不接收梯度）
            #   - 但会跟着模型移动到 GPU/MPS，也会保存进 checkpoint
            self.register_buffer("proto_sum", torch.zeros(num_classes, latent_dim))
            # prototypes = proto_sum 的方向，用于计算 loss，也供之后的控制使用。
            self.register_buffer("prototypes", torch.zeros(num_classes, latent_dim))
        else:
            raise ValueError(f"unknown prototype mode: {mode!r}")

    @torch.no_grad()
    def _update_class_mean(self, z_norm: Tensor, targets: Tensor) -> None:
        """class_mean 模式：用当前 batch 更新每个动作的平均方向。

        proto_sum[k] = momentum × proto_sum[k] + 这个 batch 里动作 k 所有点的单位向量之和
        prototypes[k] = proto_sum[k] 的方向

        - 点多的 batch 加进去的量多，影响大；点少的影响小（按点数自动加权）。
        - 某个动作在这个 batch 里没有点时，只做淡化、不加新值，方向保持不变。
        - 第一个 batch 时 proto_sum 是 0，所以直接等于第一个 batch 的方向，不需要随机初始值。
        """
        # 把每个点的单位向量按动作类别加起来，得到 (K, D)。
        batch_sum = torch.zeros_like(self.proto_sum)
        batch_sum.index_add_(0, targets, z_norm.detach())
        self.proto_sum.mul_(self.momentum).add_(batch_sum)
        # 只更新累计和不为 0 的动作（还没出现过的动作保持 0 向量）。
        seen = self.proto_sum.norm(dim=-1) > 0
        self.prototypes[seen] = F.normalize(self.proto_sum[seen], dim=-1)

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        # idleは動作方向が不安定なので除外，不包括class0安定状态
        non_idle = labels != self.idle_label

        if non_idle.sum() == 0:
            return z.sum() * 0.0

        # idle以外の潜在変数だけを取り出す
        z_active = z[non_idle]

        # idle=0、動作=1..Kを想定し、0-indexedに変換
        targets = labels[non_idle] - (1 if self.idle_label == 0 else 0)

        # ここが設計の肝：両方を単位ベクトルに正規化することで、
        # コサイン類似度（=角度のみ）で分類する
        z_norm = F.normalize(z_active, dim=-1)

        # class_mean：只在训练时用这个 batch 更新 prototype（验证/评估时固定不动）。
        # 先更新再算 loss，这样 loss 用的是包含当前 batch 的最新方向。
        if self.mode == "class_mean" and self.training:
            self._update_class_mean(z_norm, targets)

        prototype_norm = F.normalize(self.prototypes, dim=-1)

        logits = z_norm @ prototype_norm.t() / self.temperature

        return F.cross_entropy(logits, targets)

    def get_class_directions(self) -> Tensor:
        """学習されたプロトタイプ方向を単位ベクトルとして返す。
        训练完成后，取得四个已经学好的 Prototype 方向。
        学習後にGeometricDecoderへ渡して使用する。
        """
        with torch.no_grad():
            return F.normalize(self.prototypes, dim=-1)


class DistanceActivityLoss(nn.Module):
    """r_tとEMG振幅のピアソン相関を最大化する。ラベル不要。"""

    def forward(
        self,
        z: Tensor,
        emg_rms: Tensor,
        z_idle_center: Tensor | None = None,
    ) -> Tensor:
        if z_idle_center is None:
            # IdleAnchorLossを使っている場合は原点でOK。
            z_idle_center = torch.zeros(z.shape[-1], device=z.device)

        r = torch.norm(z - z_idle_center.unsqueeze(0), dim=-1)
        activity = emg_rms.reshape(-1).to(device=r.device, dtype=r.dtype)
        if activity.numel() != r.numel():
            raise ValueError(
                "emg_rms must contain one scalar per latent sample: "
                f"got {activity.numel()} values for {r.numel()} samples"
            )

        # ピアソン相関。
        r_c = r - r.mean()
        a_c = activity - activity.mean()
        corr = (r_c * a_c).sum() / (
            torch.sqrt((r_c ** 2).sum() * (a_c ** 2).sum()) + 1e-8
        )
        return 1.0 - corr
    
