"""YAML 設定から実験を回す汎用ランナー．
根据YAML配置运行真实EMG的VanillaVAE实验。
Usage::

    uv run python examples/02_run_from_yaml.py --config configs/vanilla_emg_1d.yaml
    uv run python examples/02_run_from_yaml.py --config configs/vanilla_emg_1d.yaml \
        --override model.latent_dim=16 train.epochs=30
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from vae_wwx.config.loader import apply_overrides, load_config
from vae_wwx.data import build_dataset
from vae_wwx.losses.builder import build_composite_loss
from vae_wwx.models.factory import build_vae
from vae_wwx.train.checkpoint import load_checkpoint
from vae_wwx.train.trainer import Trainer
from vae_wwx.utils.seed import set_global_seed
from vae_wwx.viz import (
    plot_latent_scatter,
    plot_loss_curves,
    plot_reconstruction_1d,
)


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(description="vae-wwx experiment runner")
    # 指定YAML配置文件。
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    # 指定需要继续训练的checkpoint。
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint から model/optimizer を読み込み，train.epochs まで継続学習する",
    )
    # 临时覆盖YAML中的配置。
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="ドット記法で上書き（例: model.latent_dim=16 train.epochs=50）",
    )
    return parser.parse_args()


def main() -> None:
    """运行完整VAE实验。"""
    # 读取命令行参数。
    args = parse_args()
    # 读取YAML配置。
    cfg = load_config(args.config)
    # 应用命令行临时配置。
    cfg = apply_overrides(cfg, args.override)
    # 新训练时创建精确到秒的独立运行目录，避免覆盖以前的实验结果。
    if args.resume is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.train.output_dir) / run_id
        # 同一秒内出现重名时直接报错，不复用或覆盖已有目录。
        run_dir.mkdir(parents=True, exist_ok=False)
        cfg.train.output_dir = str(run_dir)
    # 断点续训时继续写入 checkpoint 所在的原运行目录。
    else:
        cfg.train.output_dir = str(Path(args.resume).resolve().parent)
    # 固定随机种子。
    set_global_seed(cfg.train.seed)
    
    # 生成图片标题中使用的配置说明。
    plot_cfg = (
        f"("
        f"{cfg.model.kind} {cfg.model.backbone} | "
        f"z={cfg.model.latent_dim} | "
        f"beta={cfg.loss.kl.weight}"
        f")"
    )
    # 根据配置创建数据集。
    dataset = build_dataset(cfg.data)
    # 根据配置创建VAE模型。
    model = build_vae(cfg.model, cfg.data)
    # 根据配置创建组合损失。
    loss = build_composite_loss(cfg.loss)
    # 创建Trainer。
    trainer = Trainer(model=model, loss=loss, dataset=dataset, cfg=cfg)
    # 默认从第0轮开始训练。
    start_epoch = 0
    # 指定checkpoint时恢复训练状态。
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            trainer.model,
            trainer.optimizer,
            map_location=trainer.device,
        )
        # 取得checkpoint中记录的训练轮数。
        start_epoch = int(payload.get("config", {}).get("train", {}).get("epochs", 0))
        # 目标轮数必须大于已经训练的轮数。
        if start_epoch >= cfg.train.epochs:
            raise ValueError(
                f"resume checkpoint epoch ({start_epoch}) must be smaller than "
                f"target train.epochs ({cfg.train.epochs})"
            )
        trainer.logger.info(f"resume from {args.resume} at epoch {start_epoch}")
    # 开始训练。
    history = trainer.fit(start_epoch=start_epoch)
    # 取得实验输出目录。
    out = Path(cfg.train.output_dir)
    # 再構成サンプル
    # 取得8个验证样本，用于原项目的重建图。
    sample = trainer.get_sample_batch(n=8)
    # 切换成验证模式。
    model.eval()
    # 重建这批验证样本。
    with torch.no_grad():
        outputs = model(sample["x"])
    # 保存原项目的4样本×4通道重建图。
    plot_reconstruction_1d(
        sample["x"], 
        outputs["x_hat"], 
        out / f"recon{plot_cfg}.png", 
        title=f"{cfg.name} recon{plot_cfg}"
    )

    # 根据配置选择latent图的数据来源。
    if cfg.viz.latent_source == "val":
        latent_loader = trainer.val_loader
        latent_name = "val_"
    # 配置为all时，读取训练集和验证集的全部数据。
    else:
        latent_loader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
        )
        latent_name = "all_"
    # 潜在散布図（many-to-many では mu が (B, W, D) なので flatten）
    # 收集需要绘制的潜在变量和标签。
    zs, ys = [], []
    with torch.no_grad():
        for batch in latent_loader:
            # 将当前batch移动到模型设备。
            batch = {k: v.to(trainer.device) for k, v in batch.items()}
            # 取得VanillaVAE输出。
            out_dict = model(batch["x"])
            # 保存mu和标签。
            zs.append(out_dict["mu"].cpu())
            ys.append(batch["y"].cpu())
    # 合并并绘制潜在空间。
    if zs:
        z_all = torch.cat(zs)
        y_all = torch.cat(ys) if ys else None
        
        plot_latent_scatter(z_all,
                            y_all,
                            out / f"latent{plot_cfg}.png", 
                            title=f"{latent_name} latent{plot_cfg}")
    # 绘制训练和验证损失曲线
    plot_loss_curves(history.train,
                     history.val,
                     out / f"curves{plot_cfg}.png",
                     title=f"{latent_name} losses{plot_cfg}")
    print(f"完了．出力: {out.resolve()}")


if __name__ == "__main__":
    main()
