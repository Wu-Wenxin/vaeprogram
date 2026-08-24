"""``LossConfig`` から ``CompositeLoss`` を組み立てるビルダー．
根据LossConfig创建VAE训练需要的组合损失。"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from vae_wwx.config.schema import LossConfig
from vae_wwx.losses.composite import CompositeLoss, LossTerm
from vae_wwx.losses.kl import (
    gaussian_kl,
    gaussian_kl_with_free_bits,
)
from vae_wwx.losses.reconstruction import (
    mse_recon_loss,
)
from vae_wwx.losses.schedules import (
    ConstantSchedule,
    build_kl_schedule
)


def build_recon_term(cfg: LossConfig) -> LossTerm:
    """根据配置创建重建损失项。"""
    # 使用均方误差。
    if cfg.recon.kind == "mse":
        fn = lambda out, batch: mse_recon_loss(out["x_hat"], batch["x"])  # noqa: E731
    # 使用平均绝对误差。
    else:
        raise ValueError(f"未対応の recon kind: {cfg.recon.kind}")
    # 创建一个带名称、计算函数和权重的损失项。
    return LossTerm(name="recon", fn=fn, schedule=ConstantSchedule(value=cfg.recon.weight))


def build_kl_term(cfg: LossConfig) -> LossTerm:
    """根据配置创建KL损失项。"""
    # 取出free bits配置。
    free_bits = cfg.kl.free_bits
    def fn(out: dict[str, Any], batch: dict[str, Any]) -> Tensor:  # noqa: ARG001
        # 计算当前批次的KL损失。
        # free_bits大于0时，使用带下限的KL损失。
        if free_bits > 0.0:
            return gaussian_kl_with_free_bits(out["mu"], out["logvar"], free_bits)
        # 否则使用普通高斯KL损失。
        return gaussian_kl(out["mu"], out["logvar"])

    return LossTerm(name="kl", fn=fn, schedule=build_kl_schedule(cfg.kl))



def build_composite_loss(cfg: LossConfig) -> CompositeLoss:
    """``LossConfig`` からまとめて ``CompositeLoss`` を組み立てる．
    根据LossConfig创建完整的VAE损失
    None の項は無視する．``classified`` / GMM などモデル種別固有の損失は
    呼び出し側で ``add`` してもよい．
    """
    # 创建VAE当前需要的两个损失项。
    terms: list[LossTerm] = [build_recon_term(cfg), build_kl_term(cfg)]

    # 将多个损失项组合成一个总损失。
    return CompositeLoss(terms)
