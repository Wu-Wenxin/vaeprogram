"""
一维信号使用的 encoder．
输入 EMG：(B, C, T)
      ↓ 一维卷积
输出特征：(B, out_features)
B: 一次输入多少个样本
C: 信号通道数，eg肌肉数量
T: 每段信号的时间长度
out_features: Encoder压缩后得到的特征数量

入力: ``(B, C, T)`` / 出力: ``(B, out_features)`` の flatten 済み特徴．
潜在ヘッド（``μ, logvar`` を出す部分）は ``BaseVAE`` 側で外付け．
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from vae_wwx.models.base import BaseEncoder


def _bottleneck_length(length: int, n_blocks: int) -> int:
    """
    计算经过多次池化后，时间长度还剩多少。
    ``MaxPool1d(k=3, s=2, p=1)`` を ``n_blocks`` 回適用後の系列長.
    各段で ``L_out = ceil(L_in / 2)``．
    """
    out_len = length
    # 用 _ 是因为不关心当前是第几次，只需要重复操作
    for _ in range(n_blocks):
        out_len = (out_len + 1) // 2  # 每次池化后，长度大约减半；+1 是为了更好的处理奇数
    return out_len


class SimpleConv1dEncoder(BaseEncoder):
    """
    使用一维卷积压缩信号。
    输入 (B, C, T)
    → 多层一维卷积
    → 每层将时间长度缩短大约一半
    → 展平成二维特征
    → 输出 (B, out_features)交给后面的VAE潜在层
    
    素朴な 1D CNN encoder.

    各段は ``Conv1d(k=3, p=1) + BN + PReLU + MaxPool(k=3, s=2, p=1)``．
    ``hidden_channels`` の長さがそのまま段数になる．
    """

    def __init__(
        self,   # 当前这个Encoder对象
        in_channels: int,   # 输入通道数
        hidden_channels: list[int],   # 每层卷积的输出通道数
        length: int # 输入信号的时间长度
    ) -> None:  # 初始化方法不返回结果
        super().__init__()
        # 检查调用者有没有指定隐藏层
        if not hidden_channels:
            raise ValueError("hidden_channels が空です")
        # 创建空列表，用于存放卷积块
        blocks: list[nn.Module] = []
        # 记录当前输入通道数
        prev = in_channels
        # 遍历隐藏通道
        for h in hidden_channels:
            # 每次循环建立一个卷积块
            blocks.append(
                # nn.Sequential 表示数据按顺序经过里面的操作
                # 完成 输入
                # → Conv1d：提取特征
                # → BatchNorm：稳定数据
                # → PReLU：增加非线性
                # → MaxPool：缩短时间长度
                nn.Sequential(
                    # 负责提取信号特征：
                    # prev：输入通道数
                    # h：输出通道数
                    # kernel_size=3：每次观察相邻3个时间点
                    # padding=1：卷积前后补一个位置，使卷积后时间长度不变
                    nn.Conv1d(prev, h, kernel_size=3, padding=1),
                    # 对卷积输出进行标准化，让数据大小更稳定，通常可以让训练更容易
                    # h 必须和上一层卷积的输出通道数相同
                    nn.BatchNorm1d(h),
                    # 激活函数
                    nn.PReLU(),
                    # 负责缩短时间长度
                    # kernel_size=3：每次观察相邻3个位置
                    # stride=2：每次向前移动2个位置，因此长度大约减半
                    # padding=1：在两边补一个位置，帮助处理边界
                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                )
            )
            # 把这一层的输出通道数，记录为下一层的输入通道数
            prev = h
        # *blocks 会把列表中的卷积块展开，交给 nn.Sequential
        # 这样后面只要写self.blocks(x) 数据就会依次经过所有卷积块
        self.blocks = nn.Sequential(*blocks)
        # 计算经过所有池化层后，剩余的时间长度。
        self.bottleneck_length = _bottleneck_length(length, len(hidden_channels))
        # 取得最后一层的通道数
        self.bottleneck_channels = hidden_channels[-1]
        # 计算展平之后的特征总数
        self.out_features = self.bottleneck_channels * self.bottleneck_length

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, T) -> (B, hidden[-1], bottleneck_length)
        # 让输入依次经过前面建立的所有卷积块。
        z = self.blocks(x)
        # flatten
        # 把通道和时间两个维度展开
        # 其中1 表示，从第1维开始展平，但保留第0维的 batch
        return z.flatten(1)  # (B, out_features)