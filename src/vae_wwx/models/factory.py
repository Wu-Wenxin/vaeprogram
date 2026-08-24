"""``ModelConfig`` から VAE を組み立てるファクトリ．
根据模型配置创建Encoder、Decoder和VAE。
「if/elif の小さな分岐 + 直接生成」だけにし，Registry やプラグイン機構は使わない．
"""

from __future__ import annotations

from vae_wwx.config.schema import DataConfig, ModelConfig
from vae_wwx.models.base import BaseDecoder, BaseEncoder, BaseVAE
from vae_wwx.models.decoders_1d import SimpleConv1dDecoder
from vae_wwx.models.encoders_1d import SimpleConv1dEncoder
from vae_wwx.models.vae_vanilla import VanillaVAE


def build_encoder_decoder(model_cfg: ModelConfig, data_cfg: DataConfig) -> tuple[BaseEncoder, BaseDecoder]:
    """``backbone`` に応じて encoder / decoder を返す．
    根据配置创建一维卷积Encoder和Decoder。"""
    if model_cfg.backbone == "simple1d":
        # Encoder把原始EMG压缩成特征。
        enc = SimpleConv1dEncoder(
            in_channels=data_cfg.channels,
            hidden_channels=model_cfg.hidden_channels,
            length=data_cfg.length,
        )
        # Decoder根据潜在变量重建EMG。
        dec = SimpleConv1dDecoder(
            latent_dim=model_cfg.latent_dim,
            hidden_channels=model_cfg.hidden_channels,
            out_channels=data_cfg.channels,
            length=data_cfg.length,
            bottleneck_length=enc.bottleneck_length,
        )
        return enc, dec
    raise ValueError(f"未対応の backbone: {model_cfg.backbone}")


def build_vae(model_cfg: ModelConfig, data_cfg: DataConfig) -> BaseVAE:
    """``ModelConfig.kind`` に応じた VAE を返す．
    根据模型配置创建完整VAE。"""
    if model_cfg.kind == "vanilla":
        enc, dec = build_encoder_decoder(model_cfg, data_cfg)
        return VanillaVAE(encoder=enc, decoder=dec, latent_dim=model_cfg.latent_dim)
    raise ValueError(f"未対応の kind: {model_cfg.kind}")
