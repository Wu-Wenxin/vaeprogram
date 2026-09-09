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
    """
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        idle_label: int = 0,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()

        self.idle_label = idle_label
        self.temperature = temperature
        # 学習可能なプロトタイプ方向。
        # forward内で単位ベクトルに正規化する。
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, latent_dim)
        )

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
    
