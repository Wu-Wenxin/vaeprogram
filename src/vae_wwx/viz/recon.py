"""再構成サンプルの可視化．
绘制一维信号的重建对比图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# 使用不需要打开图形窗口的绘图后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import Tensor

from matplotlib.lines import Line2D


def plot_reconstruction_1d(
    x: Tensor,
    x_hat: Tensor,
    save_path: str | Path,
    n_samples: int = 4,
    n_channels_show: int = 4,
    title: str = "Reconstruction (1D)",
) -> None:
    """1D 信号の再構成を ``(サンプル × チャネル)`` グリッドで描画．
    按照样本数×通道数绘制原始信号和重建信号。
    入力は ``(B, C, T)``．``B>=n_samples``，``C>=n_channels_show`` を仮定．
    """
    # 将保存位置转换成Path。
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 从计算图分离并转换成NumPy数组。
    x_np = x.detach().cpu().numpy()
    x_hat_np = x_hat.detach().cpu().numpy()

    # 防止请求的样本数超过实际batch大小。
    n_samples = min(
        n_samples,
        x_np.shape[0],
    )

    # 防止请求的通道数超过实际通道数量。
    n_channels_show = min(
        n_channels_show,
        x_np.shape[1],
    )

    # 创建“样本数×通道数”的子图。
    fig, axes = plt.subplots(
        n_samples,
        n_channels_show,
        figsize=(
            2.75 * n_channels_show,
            1.35 * n_samples,
        ),
        squeeze=False,
        sharex=True,
    )

    # 依次绘制每个样本的每个通道。
    for i in range(n_samples):
        for c in range(n_channels_show):
            # 取得当前位置的子图。
            ax = axes[i][c]

            # 黑色曲线表示原始EMG。
            ax.plot(
                x_np[i, c],
                color="black",
                linewidth=0.8,
                label=(
                    "x"
                    if (i, c) == (0, 0)
                    else None
                ),
            )

            # 红色曲线表示VAE重建EMG。
            ax.plot(
                x_hat_np[i, c],
                color="tab:red",
                linewidth=0.8,
                label=(
                    "x_hat"
                    if (i, c) == (0, 0)
                    else None
                ),
            )

            # 第一行显示通道名称。
            if i == 0:
                ax.set_title(
                    f"ch{c}",
                    fontsize=9,
                )

            # 设置坐标刻度字体大小。
            ax.tick_params(labelsize=9)

    # 设置整张图片标题。
    fig.suptitle(
        title,
        fontsize=11,
    )

    # 设置公共横轴名称。
    fig.supxlabel(
        "Time (samples)",
        fontsize=9,
        y=0.02,
    )

    # 设置公共纵轴名称。
    fig.supylabel(
        "Standardized EMG amplitude (z-score)",
        fontsize=9,
        x=0.012,
    )

    # 显示原始信号和重建信号图例。
    fig.legend(
        loc="upper right",
        fontsize=8,
    )

    # 调整子图间距和页面边距。
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.095,
        top=0.86,
        wspace=0.28,
        hspace=0.36,
    )

    # 保存图片。
    fig.savefig(
        save_path,
        dpi=120,
    )

    # 关闭图片，释放内存。
    plt.close(fig)
    
    
def plot_reconstruction_channels_overlay(
    x: Tensor,
    x_hat: Tensor,
    save_path: str | Path,
    sample_index: int = 0,
    n_channels_show: int = 8,
    title: str = "8-Channel Original vs Reconstruction",
) -> None:
    """在同一个坐标系中叠加显示多通道原始和重建信号。"""

    # 创建图片保存目录。
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换成NumPy数组。
    x_np = x.detach().cpu().numpy()
    x_hat_np = x_hat.detach().cpu().numpy()

    # 限制实际显示的通道数量。
    n_channels_show = min(
        n_channels_show,
        x_np.shape[1],
    )

    # 创建一个坐标系。
    fig, ax = plt.subplots(figsize=(12, 6))

    # 使用tab10中的不同颜色区分通道。
    colors = plt.get_cmap("tab10").colors

    # 绘制每个通道的原始和重建信号。
    for channel in range(n_channels_show):
        color = colors[channel]

        # 实线表示原始信号。
        ax.plot(
            x_np[sample_index, channel],
            color=color,
            linestyle="-",
            linewidth=1.0,
            alpha=0.9,
        )

        # 同色虚线表示对应的重建信号。
        ax.plot(
            x_hat_np[sample_index, channel],
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )

    # 创建通道颜色图例。
    channel_handles = [
        Line2D(
            [0],
            [0],
            color=colors[channel],
            linewidth=2,
            label=f"CH{channel + 1}",
        )
        for channel in range(n_channels_show)
    ]

    channel_legend = ax.legend(
        handles=channel_handles,
        title="Channels",
        loc="upper left",
        ncol=4,
        fontsize=8,
    )

    # 保留第一个图例。
    ax.add_artist(channel_legend)

    # 创建原始和重建线型图例。
    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            label="Original x",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label="Reconstruction x_hat",
        ),
    ]

    ax.legend(
        handles=style_handles,
        title="Line style",
        loc="upper right",
        fontsize=8,
    )

    # 设置标题、坐标名称和网格。
    ax.set_title(title)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel(
        "Standardized EMG amplitude (z-score)"
    )
    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    # 调整布局并保存图片。
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)