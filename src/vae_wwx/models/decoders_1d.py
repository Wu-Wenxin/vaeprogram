"""1D 信号用 decoder．
潜在变量 z：(B, latent_dim)
→ 全连接层
→ 恢复成三维数据
→ 多层转置卷积
→ 重建信号：(B, C, T)
encoder と対称に組む．``ConvTranspose1d(k=3, s=2, p=1, op=1)`` で時間長を倍々に戻し，
最後に ``F.interpolate`` で原系列長へきっちり合わせる（端数があっても破綻しない）．
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vae_wwx.models.base import BaseDecoder


class SimpleConv1dDecoder(BaseDecoder):
    """``SimpleConv1dEncoder`` のミラー decoder.

    Args:
        latent_dim: 潜在次元 D．
        hidden_channels: encoder と同じリスト（内部で逆順に展開）．
        out_channels: 出力チャネル数 C（入力と一致させる）．
        length: 出力系列長 T（最終 ``F.interpolate`` でこの長さに合わせる）．
        bottleneck_length: encoder の ``bottleneck_length``．
    """

    def __init__(
        self,
        latent_dim: int,  # 潜在变量z 有几个数字
        hidden_channels: list[int],  # 与Encoder 使用相同的通道列表
        out_channels: int,  # 最终重建信号的通道数
        length: int,  # 最终重建信号的时间长度
        bottleneck_length: int,  # Encoder 压缩后的时间长度
    ) -> None:
        super().__init__()
        if not hidden_channels:
            raise ValueError("hidden_channels が空です")
        # 把Encoder 的通道顺序反过来
        rev = list(reversed(hidden_channels))  # [h_last, ..., h_first]
        # 记录Decoder 开始时的通道数
        self.bottleneck_channels = rev[0]
        # 记录Encoder 压缩后的时间长度
        self.bottleneck_length = bottleneck_length
        # 记录最终需要还原到的长度
        self.target_length = length
        # 创建全连接成，把很短的潜在变量展开
        self.fc = nn.Linear(latent_dim, rev[0] * bottleneck_length)

        # チャネル列 [h_last, ..., h_first, out_channels] を順に ConvTranspose1d で繋ぐ．
        # 計 len(hidden_channels) 段で encoder の MaxPool 回数と一致させる．
        chs = rev + [out_channels]
        # 创建空列表，用于保存每个转置卷积块
        blocks: list[nn.Module] = []
        # 遍历相邻通道
        for i in range(len(chs) - 1):
            in_c, out_c = chs[i], chs[i + 1]
            # 判断当前是不是最后一层
            # 因为最后一层直接输出重建信号，不再添加 BatchNorm 和激活函数
            is_last = i == len(chs) - 2
            # 创建一个转置卷积块
            block: list[nn.Module] = [
                nn.ConvTranspose1d(
                    in_c,  # 输入通道数
                    out_c,  # 输出通道数
                    kernel_size=3,  # 每次处理相邻3个位置
                    stride=2,  # 时间长度扩大约2倍
                    # 让输出长度正好扩大2倍
                    padding=1,
                    output_padding=1
                    ),
            ]
            # 不是最后一层的情况下添加和激活函数
            # 中间层：转置卷积 + BatchNorm + PReLU
            # 最后层：只使用转置卷积
            if not is_last:
                # 增加中间数据
                block.append(nn.BatchNorm1d(out_c))
                # 增加非线形
                block.append(nn.PReLU())
            # 把当前层组合起来，再加入总列表
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z: Tensor) -> Tensor:
        # (B, D) -> (B, h_last * bottleneck_length)
        h = self.fc(z)
        # (B, h_last, bottleneck_length)
        # 恢复成三维数据
        h = h.view(-1, self.bottleneck_channels, self.bottleneck_length)
        # 段階的に T を倍々で戻す：(B, h_last, T_b) -> (B, C_out, T_b * 2^N)
        h = self.blocks(h)
        # 端数対応: 想定 T と一致しない場合のみ補正
        # 如果时间长度的实际长度不等于目标长度，就进行修正
        if h.size(-1) != self.target_length:
            h = F.interpolate(h, size=self.target_length, mode="linear", align_corners=False)
        return h