# 规定所有模型的基本接口
# 只规定 Encoder、Decoder 和 VAE 必须提供哪些方法，不负责真正计算。

# 让 Python 暂时不立即解析类型标注。
from __future__ import annotations
# 倒入抽象类工具，ABC 表示这个类只是基础模版
# abstractmethod 表示表示这个方法必须由后面的具体模型完成
from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module, ABC):
    """
    所有 Encoder 的共同规则。
    ``forward`` は flatten 済みの特徴ベクトル ``(B, out_features)`` を返す．
    """
    # 说明压缩后有多少个特征
    out_features: int
    # BaseEncoder 只要求有 forward()，但不决定里面使用几层卷积。
    # 没有写具体计算，因为卷积计算会在 encoders_1d.py 里实现。
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class BaseDecoder(nn.Module, ABC):
    """所有 Decoder 的共同规则。
    接收潜在变量 z → 还原出数据
    具体如何还原，由 decoders_1d.py 决定。
    ``forward(z)`` は元入力と同じ形状のテンソルを返す．"""

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor:
        pass


class BaseVAE(nn.Module, ABC):
    """所有 VAE 的共同规则。
    规定所有 VAE 必须有三个操作。
    - 编码``encode(x, **cond) -> dict``: ``mu`` / ``logvar`` 等の事後パラメータ
    - 解码``decode(z, **cond) -> x_hat``
    - 完整运行``forward(x, **cond) -> dict``: ``encode`` + ``reparameterize`` + ``decode`` をまとめる
    x → encode → z → decode → x_hat
    **cond 是可选的标签等额外条件，让这个接口也能支持 Conditional VAE
    継承クラスは **単一継承 + 組成** で書く（多重継承の塔を避ける）．
    """

    @abstractmethod
    def encode(self, x: Tensor, **cond: Tensor) -> dict[str, Tensor]:
        pass

    @abstractmethod
    def decode(self, z: Tensor, **cond: Tensor) -> Tensor:
        pass

    @abstractmethod
    def forward(self, x: Tensor, **cond: Tensor) -> dict[str, Tensor]:
        pass