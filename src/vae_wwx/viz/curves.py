"""損失曲線と重みスケジュール推移の描画．
绘制损失曲线和损失权重变化。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(
    train_history: list[dict[str, float]],
    val_history: list[dict[str, float]] | None,
    save_path: str | Path,
    keys: list[str] | None = None,
    title: str = "Loss curves",
) -> None:
    """エポック単位の損失曲線を描画.
    按照epoch绘制训练损失和验证损失。
    ``train_history`` の各要素は少なくとも ``epoch`` キーを持ち，それ以外を全部プロット．
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 没有训练记录时不画图。
    if not train_history:
        return
    # 自动找出需要绘制的损失，排除epoch和权重。
    auto_keys = [k for k in train_history[0] if k not in {"epoch"} and not k.startswith("w_")]
    keys = keys or auto_keys
    # 取得所有epoch。
    epochs = [h["epoch"] for h in train_history]

    # 每一种损失使用一个子图。
    fig, axes = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 3), squeeze=False)
    # 分别绘制total、recon和kl等损失。
    for i, k in enumerate(keys):
        ax = axes[0][i]
        # 取得训练损失。
        train_y = [h.get(k, float("nan")) for h in train_history]
        ax.plot(epochs, train_y, label="train", color="tab:blue")
        # 存在验证记录时绘制验证损失。
        if val_history:
            val_y = [h.get(k, float("nan")) for h in val_history]
            ax.plot(epochs, val_y, label="val", color="tab:orange", linestyle="--")
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Loss (epoch average)")
        ax.tick_params(labelsize=8)
        # 只在第一个子图显示图例。
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_weight_schedule(
    train_history: list[dict[str, float]],
    save_path: str | Path,
    title: str = "Loss component weights",
) -> None:
    """``w_<name>`` キーで記録された重み推移を可視化．
    绘制以w_开头的损失权重变化。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not train_history:
        return
    # 找出w_recon和w_kl等权重记录。
    weight_keys = [k for k in train_history[0] if k.startswith("w_")]
    if not weight_keys:
        return
    # 取得所有epoch。
    epochs = [h["epoch"] for h in train_history]
    # 创建一张权重曲线图。
    fig, ax = plt.subplots(figsize=(4.5, 3))
    # 分别绘制每一种损失权重。
    for k in weight_keys:
        ys = [h.get(k, float("nan")) for h in train_history]
        ax.plot(epochs, ys, label=k.removeprefix("w_"))
    ax.set_xlabel("epoch")
    ax.set_ylabel("weight (epoch avg)")
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
