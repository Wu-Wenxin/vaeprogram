"""再構成損失．
计算重构信号和原始信号之间的误差
すべて ``(x_hat, x) -> scalar`` のシンプルな関数．返り値は **per-sample で
データ次元方向に sum，バッチ方向に mean** したもの．

**スケール規約**:
    Kingma & Welling, 2014 の ELBO はサンプルごとに

        log p(x|z) = Σ_{d=1..D} log p(x_d|z)
        KL         = Σ_{k=1..D_z} (1/2)(μ_k² + σ_k² − 1 − log σ_k²)

    と「データ次元の和 − 潜在次元の和」で書かれる．これに合わせて
    再構成損失もデータ次元方向に **sum** を取り，KL も ``sum(dim=D_z)`` を採用する．
    両方を sum で揃えるので β=1.0 が ELBO 最大化に対応する canonical default になる．

    ``F.mse_loss(reduction="mean")`` のように全要素平均にすると，暗黙のうちに
    recon が ``1/(C·T)`` 倍に縮み，β を ``C·T`` 倍程度小さく取らないと
    posterior collapse しやすい．
"""

from __future__ import annotations

from torch import Tensor

# 分别计算每个样本的总误差
def _sum_per_sample(value: Tensor) -> Tensor:
    """``(B, …)`` 形式のテンソルをサンプルごとに flatten し，最終次元方向に和を取る．"""
    # (B, C, T) → (B, C × T)，保留第0维，从第1维开始，把后面的所有维度压平成一维
    # .sum(dim=-1)，(B, C × T) → (B,)
    return value.flatten(1).sum(dim=-1)

# MSE重构损失
def mse_recon_loss(x_hat: Tensor, x: Tensor) -> Tensor:
    """``Σ_d (x_hat - x)²`` の per-sample sum をバッチ平均.

    σ²=1/2 のガウス観測モデル ``N(x_hat, σ²I)`` の NLL（定数項を除く）と等価．
    σ² は ``LossConfig.recon.weight`` を変えることで実質的に調整できる
    （重み 0.5 倍 ≡ σ²=1 を仮定したのと同じ）．
    """
    # 得到每个位置的误差，然后进行平方，这样就可以避免正负误差互相抵消
    sq = (x_hat - x).pow(2)
    # 计算整个 batch 的平均误差，得到一个数字
    # 数字越小，说明重构信号越接近原始信号
    return _sum_per_sample(sq).mean()