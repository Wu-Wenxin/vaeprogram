"""
Encoder 输出特征
→ 生成 mu 和 logvar
→ 随机采样得到 z
→ 交给 Decoder

潜在空間と分類のヘッド群．"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    根据mu 和logvar 生成潜在变量z
    再パラメータ化トリック: ``z = μ + σ ⊙ ε``，``ε ~ N(0, I)``.

    ``logvar`` は ``log σ²``（logの方が学習が安定）．
    """
    # 把logvar 转换成标准差std
    std = torch.exp(0.5 * logvar)
    # 生成一组形状相同的标准正态随机数
    eps = torch.randn_like(std)
    # 根据中心mu 和标准差std 得到随机的z
    # z = 中心位置 + 随机偏移
    return mu + eps * std


class GaussianLatentHead(nn.Module):
    """
    把Encoder 特征转换成mu 和logvar
    mu：潜在分布的中心位置
    logvar：潜在分布的方差大小，准确说是方差的对数
    VAE 时显得到一个分步：mu + logvar → 一个潜在分布 → 从中采样 z
    flatten 済み特徴 → ``(μ, logvar)`` の線形ヘッド.

    Args:
        in_features: 入力特徴次元（encoder の ``out_features``）．
        latent_dim: 潜在次元 D．
    """

    def __init__(
        self,
        in_features: int,
        latent_dim: int
    ) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        mu = self.fc_mu(feat)        # (B, D)
        logvar = self.fc_logvar(feat)  # (B, D)
        # logvar が暴走しないよう緩く clamp（学習初期の数値安定）
        logvar = torch.clamp(logvar, min=-8.0, max=8.0)
        return mu, logvar