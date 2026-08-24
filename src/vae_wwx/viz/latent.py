"""潜在空間の散布図．高次元は PCA で 2 次元に投影．"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def _pca2(z: np.ndarray) -> np.ndarray:
    """素朴な PCA で先頭 2 軸へ射影．``z`` は ``(N, D)``，戻り値は ``(N, 2)``．
    将形状为(N, D)的潜在变量转换成(N, 2)。"""
    # 潜在维度小于或等于2时不需要PCA。
    if z.shape[1] <= 2:
        # 创建固定为二维的空数组。
        out = np.zeros((z.shape[0], 2), dtype=z.dtype)
        # 将原来的潜在变量放入前面的维度。
        out[:, : z.shape[1]] = z
        return out
    # 每个潜在维度减去平均值。
    centered = z - z.mean(axis=0, keepdims=True)
    # 共分散の SVD（NumPyのみで実装：次元低い前提）
    # 使用SVD计算PCA方向。
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    # 投影到最主要的两个方向。
    return centered @ vt[:2].T


def plot_latent_scatter(
    z: Tensor,
    labels: Tensor | None,
    save_path: str | Path,
    title: str = "Latent space",
) -> None:
    """``(N, D)`` の潜在ベクトルをラベル別に色分けして散布．D>2 は PCA で射影．
    按照标签颜色绘制潜在变量散点图。"""
    # 转换并创建图片保存路径。
    save_path = Path(save_path)
    # 将Tensor转换成NumPy数组。
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 将潜在变量转换成二维坐标。
    z_np = z.detach().cpu().numpy()
    # latent_dim==1 でも (N, 2) になるよう常に通す．_pca2 は D<=2 をゼロパッドする．
    # 标题较长时自动增加图片宽度。
    z2 = _pca2(z_np)

    # 标题较长时自动增加图片宽度。
    fig_width = max(5.0, min(12.0, 5.0 + 0.07 * max(len(title) - 35, 0)))
    # 创建散点图。
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    # 没有标签时使用同一种颜色。
    if labels is None:
        ax.scatter(z2[:, 0], z2[:, 1], s=8, alpha=0.6)
    # 存在标签时按照类别分别绘制。
    else:
        # 将标签转换成NumPy数组。
        y = labels.detach().cpu().numpy()
        # 依次绘制每一个类别。
        for cls in np.unique(y):
            # 找出属于当前类别的样本。
            sel = y == cls
            # 绘制当前类别的潜在变量。
            ax.scatter(z2[sel, 0], z2[sel, 1], s=8, alpha=0.6, label=f"class {int(cls)}")
        # 显示类别图例。
        ax.legend(fontsize=8)
    # 设置标题和坐标名称。
    ax.set_title(title)
    ax.set_xlabel("z[0] (or PC1)")
    ax.set_ylabel("z[1] (or PC2)")
    ax.tick_params(labelsize=8)
    # 调整布局并保存图片。
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
