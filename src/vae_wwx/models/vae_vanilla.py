"""Vanilla VAE．
x
→ encode()
→ mu、logvar
→ reparameterize()
→ z
→ decode()
→ x_hat
最も単純な構成．拡張系（Conditional / Classified / GMM / Sequence）はこのクラスを
継承せず独立クラスとして書く（多重継承の塔を避けるため）．
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from vae_wwx.models.base import BaseDecoder, BaseEncoder, BaseVAE
from vae_wwx.models.heads import GaussianLatentHead, reparameterize


class VanillaVAE(BaseVAE):
    """Vanilla Gaussian VAE．

    Args:
        encoder: backbone（``out_features`` を持つ）．
        decoder: backbone（``z`` から元入力形へ）．
        latent_dim: 潜在次元 D．
    """

    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder, latent_dim: int) -> None:
        super().__init__()
        # 创建潜在层
        self.encoder = encoder
        self.decoder = decoder
        self.latent_head: nn.Module = GaussianLatentHead(encoder.out_features, latent_dim)
        self.latent_dim = latent_dim

    # 把输入信号转换成潜在分布参数
    def encode(
        self,
        x: Tensor,
        **cond: Tensor
    ) -> dict[str, Tensor]:  # noqa: ARG002
        # (B, ...) -> (B, out_features) -> (μ, logvar)
        features = self.encoder(x)
        mu, logvar = self.latent_head(features)
        return {
            "mu": mu,
            "logvar": logvar
            }

    def decode(self, z: Tensor, **cond: Tensor) -> Tensor:  # noqa: ARG002
        # 因为是最简单的vae 没有任何附加的条件，所以直接返回即可
        return self.decoder(z)

    def forward(self, x: Tensor, **cond: Tensor) -> dict[str, Tensor]:  # noqa: ARG002
        params = self.encode(x)
        # 在潜在分布中采样z
        z = reparameterize(params["mu"], params["logvar"])
        # 根据z 重构输入信号
        x_hat = self.decode(z)
        return {
            "x_hat": x_hat,
            "z": z, **params
            }
