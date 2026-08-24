"""損失の重み付き合成．
把多个损失按照不同权重加在一起
负责把两个损失相加起来 总损失 = 重建损失 + β × KL损失
重建损失：要求 x_hat 接近 x
KL损失：要求潜在空间保持规整
β：控制 KL 损失的重要程度

「損失成分 + その重みスケジュール」を任意個まとめて1つのスカラーへ合算する．
VAE 固有の構造に依存しないインターフェース．
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from vae_wwx.losses.schedules import ConstantSchedule, Schedule

# 定义损失函数的统一格式
# (model_outputs, batch) -> scalar Tensor
# 表示每个损失函数都应该 输入：模型输出outputs, 当前数据batch
# 输出：一个损失Tensor
LossFn = Callable[
    [dict[str, Any], dict[str, Any]],
    Tensor
    ]


@dataclass
class LossTerm:
    """1つの損失成分．
    定义一个损失项，保存一个损失函数及其权重变化的方式
    Attributes:
        name: 表示・ログ用名前．
        fn: 損失関数 ``(outputs, batch) -> scalar``．
        schedule: ``step -> weight`` のスケジュール（``None`` なら重み1 固定）．
    """

    name: str  # 损失名称
    fn: LossFn  # 真正计算损失的函数
    schedule: Schedule | None = None  # 控制这个损失当前的权重

    # 取得当前权重的方法
    def weight_at(self, step: int) -> float:
        if self.schedule is None:
            return 1.0  # 默认权重
        return self.schedule(step)


class CompositeLoss:
    """複数の ``LossTerm`` を重み付き合算．
    把多个损失项加在一起
    这个VAE有俩损失：1. 重建损失；2. KL损失
    ``step(outputs, batch, global_step)`` で総損失と内訳を返す．
    """

    def __init__(self, terms: list[LossTerm]) -> None:
        self.terms = terms

    def __call__(
        self, 
        outputs: dict[str, Any], 
        batch: dict[str, Any], 
        global_step: int
    ) -> tuple[Tensor, dict[str, float], dict[str, float]]:
        """損失とログ用辞書を返す.

        Returns:
            ``(total_loss, components_value, weights)``．
            ``components_value[name]`` は重み**前**の生の値（解釈用）．
        """
        # 初始化。表示目前还没有任何累加损失
        total: Tensor | None = None
        values: dict[str, float] = {}  # 保存损失值
        weights: dict[str, float] = {}  # 保存权重
        # 遍历每一个损失项
        for term in self.terms:
            raw = term.fn(outputs, batch)
            w = term.weight_at(global_step)
            weighted = w * raw
            # total = weighted if total is None else total + weighted
            # 等同于
            if total is None:
                total = weighted
            else:
                total = total + weighted
            # raw.detach().item() 表示把损失从用于训练的 Tensor，转换成普通 Python 数字，方便打印和保存日志。它不会用于反向传播。
            values[term.name] = float(raw.detach().item())
            weights[term.name] = float(w)
        if total is None:
            total = torch.tensor(0.0)
        return total, values, weights

    def add(self, term: LossTerm) -> None:
        """損失成分を後から追加．
        训练前可以继续加入新的损失
        不修改 CompositeLoss 内部计算代码，也能增加新的损失项"""
        self.terms.append(term)


def constant(value: float) -> Schedule:
    """関数的に ``ConstantSchedule`` を作るショートカット．
    方便创建固定权重的方法
    只是把ConstantSchedule(value=1.0)简化成constant(value=1.0)"""
    return ConstantSchedule(value=value)
