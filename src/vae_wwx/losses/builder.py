"""``LossConfig`` から ``CompositeLoss`` を組み立てるビルダー．
根据LossConfig创建VAE训练需要的组合损失。"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from vae_wwx.config.schema import LossConfig
from vae_wwx.losses.composite import CompositeLoss, LossTerm
from vae_wwx.losses.geometric import (
    AngularPrototypeLoss,
    DistanceActivityLoss,
    IdleAnchorLoss,
)
from vae_wwx.losses.kl import (
    gaussian_kl,
    gaussian_kl_with_free_bits,
)
from vae_wwx.losses.reconstruction import (
    mse_recon_loss,
)
from vae_wwx.losses.schedules import (
    ConstantSchedule,
    build_kl_schedule,
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


def build_idle_anchor_term(cfg: LossConfig) -> LossTerm:
    """Idle Anchor Lossを作成する。"""

    idle_loss = IdleAnchorLoss(idle_label=0)
    latent_input = cfg.idle_anchor.input
    if latent_input not in ("mu", "z"):
        raise ValueError(
            f"idle_anchor.input must be 'mu' or 'z', got {latent_input!r}"
        )

    def fn(
        out: dict[str, Any],
        batch: dict[str, Any],
    ) -> Tensor:
        return idle_loss(out[latent_input], batch["y"])

    return LossTerm(
        name="idle_anchor",
        fn=fn,
        schedule=ConstantSchedule(
            value=cfg.idle_anchor.weight
        ),
    )


def build_angular_prototype_term(
    cfg: LossConfig,
    angular_loss: AngularPrototypeLoss,
) -> LossTerm:
    """Angular Prototype Lossを作成する。"""

    latent_input = cfg.angular_prototype.input
    if latent_input not in ("mu", "z"):
        raise ValueError(
            f"angular_prototype.input must be 'mu' or 'z', got {latent_input!r}"
        )

    def fn(
        out: dict[str, Any],
        batch: dict[str, Any],
    ) -> Tensor:
        return angular_loss(out[latent_input], batch["y"])

    return LossTerm(
        name="angular_prototype",
        fn=fn,
        schedule=ConstantSchedule(
            value=cfg.angular_prototype.weight
        ),
    )


def build_distance_activity_term(cfg: LossConfig) -> LossTerm | None:
    """潜在半径と標準化前EMG RMSの相関損失を作成する。"""

    # 权重为0时不构建、也不计算该损失。
    if cfg.distance_activity.weight == 0.0:
        return None
    distance_loss = DistanceActivityLoss()
    latent_input = cfg.distance_activity.input
    if latent_input not in ("mu", "z"):
        raise ValueError(
            f"distance_activity.input must be 'mu' or 'z', got {latent_input!r}"
        )

    def fn(out: dict[str, Any], batch: dict[str, Any]) -> Tensor:
        # RealEMGDatasetは標準化前RMSを返す。他Datasetでは入力xから代替計算する。
        emg_rms = batch.get("emg_rms")
        if emg_rms is None:
            emg_rms = batch["x"].flatten(1).square().mean(dim=1).sqrt()
        return distance_loss(out[latent_input], emg_rms)

    return LossTerm(
        name="distance_activity",
        fn=fn,
        schedule=ConstantSchedule(value=cfg.distance_activity.weight),
    )


def build_composite_loss(
    cfg: LossConfig,
    angular_loss: AngularPrototypeLoss | None = None,
) -> CompositeLoss:
    """``LossConfig`` からまとめて ``CompositeLoss`` を組み立てる．
    根据LossConfig创建完整的VAE损失
    None の項は無視する．``classified`` / GMM などモデル種別固有の損失は
    呼び出し側で ``add`` してもよい．
    """
    # 创建VAE当前需要的两个损失项。
    terms: list[LossTerm] = [
        build_recon_term(cfg),
        build_kl_term(cfg),
        build_idle_anchor_term(cfg),
    ]
    distance_activity_term = build_distance_activity_term(cfg)
    if distance_activity_term is not None:
        terms.append(distance_activity_term)
    if angular_loss is not None:
        terms.append(
            build_angular_prototype_term(
                cfg,
                angular_loss,
            )
        )
    # 将多个损失项组合成一个总损失。
    return CompositeLoss(terms)
