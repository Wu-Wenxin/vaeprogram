"""KL ダイバージェンス．
计算潜在分布与标准正态分布之间的差距

標準正規 prior に対する解析KL，free-bits版，GMM prior 版を提供．

解析KLの導出は ``docs/theory.md`` を参照（``q(z|x)=N(μ, σ²I)``，
``p(z)=N(0, I)`` の場合）．
"""

from __future__ import annotations

import torch
from torch import Tensor

# 检查模型得到的潜在分布，和标准正态分布 N(0, 1) 相差多少
def gaussian_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    """``KL[N(μ, σ²) || N(0, I)]`` の解析解（per-sample sum，バッチ mean）.

    ``KL = 0.5 * Σ_d (μ² + σ² - 1 - log σ²)``

    入力は ``(B, D)`` でも ``(B, ..., D)`` でも良い．サンプル軸 (dim=0) 以外を
    全部和にして per-sample 値とし，バッチ方向に平均する（系列 VAE の m2m が
    ``(B, W, D)`` を渡す場合も自然に ``Σ_w KL_w / B`` となる）．
    """
    # mu 偏离0越远，KL通常越大；反差偏移1越远，KL通常越大
    # KL越小，潜在空间越规整
    kl_elem = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    # 每个样本的所有潜在维度求和 → 再对整个 batch 求平均 → 返回一个损失数字
    return kl_elem.flatten(1).sum(dim=-1).mean()

def gaussian_kl_with_free_bits(mu: Tensor, logvar: Tensor, free_bits: float) -> Tensor:
    """潜在次元ごとに下限 ``free_bits`` (nats) を設けた KL.
    为每个潜在维度设置free-bits下限。
    ``free_bits > 0`` のとき，各潜在次元の KL 寄与が ``free_bits`` を下回る間は
    勾配を流さない（posterior collapse 抑制）．``(B, ..., D)`` 入力では，
    サンプル/window を平均してから D 方向に clamp する（free-bits は潜在次元 D 単位）．
    """
    # 计算每个样本、每个潜在维度的KL。
    kl_elem = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # (B, ..., D)
    # free_bits关闭时，结果与普通KL相同。
    if free_bits <= 0.0:
        return kl_elem.flatten(1).sum(dim=-1).mean()
    # 对所有样本求平均，保留潜在维度D。
    kl_dim = kl_elem.reshape(-1, kl_elem.size(-1)).mean(dim=0)  # (D,)
    # KL低于free_bits时停止继续向下压缩。
    capped = torch.clamp(kl_dim, min=free_bits)
    # 将所有潜在维度的KL相加。
    return capped.sum()
