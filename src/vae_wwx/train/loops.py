"""学習ループ（純関数版）．
负责真正训练模型
输入数据
→ 模型计算
→ 计算损失
→ 反向传播
→ 更新模型参数
``Trainer`` から呼び出されるが，関数単独でも使える．
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader


def _move(batch: dict[str, Any],
        device: torch.device) -> dict[str, Any]:
    # 根据旧字典，快速生成一个新字典
    # return {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
    # 等价于
    new_batch = {}
    for key, value in batch.items():
        # 判断这个value 是不是Tensor 类型
        if isinstance(value, Tensor):
            new_batch[key] = value.to(device)
        else:
            new_batch[key] = value
    return new_batch

def train_epoch(
    model: torch.nn.Module,
    loss_fn: Callable[[dict, dict, int], tuple[Tensor, dict, dict]],  # 计算总损失
    optimizer: torch.optim.Optimizer,  # 根据梯度更新模型参数
    loader: DataLoader,  # 分批提供训练数据
    device: torch.device,  # 使用CPU、GPU或MPS
    global_step: int,  # CompositeLoss 用它查询当前损失权重
    grad_clip: float | None = None,  # 防止梯度突然变得过大
) -> tuple[dict[str, float], int]:
    """1 エポック学習．

    Args:
        model: ``forward(x, **cond)`` を持つ ``BaseVAE``．
        loss_fn: ``CompositeLoss`` 互換 ``(outputs, batch, step) -> (total, values, weights)``．
        optimizer: PyTorch optimizer．
        loader: バッチが ``{"x": Tensor, "y": Tensor}`` 等の辞書．
        device: 学習デバイス．
        global_step: エポック開始時のグローバルステップ．
        grad_clip: ``None`` でクリップしない．
        cond_keys: 条件として model に渡すバッチキー（例 ``("y",)`` ）．

    Returns:
        返回更新后的global_step
        ``(epoch_metrics, new_global_step)``．
    """
    # 现在进入训练
    model.train()
    # 所有batch的损失总和
    sums: dict[str, float] = {}
    # 一共训练了多少个batch
    batch_counts = 0
    for batch in loader:
        # 把当前 batch 中的张量移动到对应设备
        batch = _move(batch, device)
        # 取出输入信号 x，送进 VAE，得到重建结果和潜在变量
        outputs = model(batch["x"])
        # 计算总损失、各项原始损失，以及各项损失当前的权重
        total, values, weights = loss_fn(outputs, batch, global_step)
        """清除旧梯度
        → 计算新梯度
        → 限制过大的梯度
        → 更新模型参数
        PyTorch 默认会累加梯度。如果不清除，这个 batch 的梯度会和上一个 batch 混在一起"""
        # 清除上一个 batch 留下的梯度
        optimizer.zero_grad(set_to_none=True)
        # 根据总损失，计算每个模型参数的梯度
        total.backward()
        # 如果设置了梯度裁剪，就限制梯度的最大值，防止训练突然不稳定
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )
        # 根据刚刚计算的梯度，更新模型参数
        optimizer.step()
        # 把当前 batch 的总损失转换成普通数字，并累加起来
        sums["total"] = sums.get("total", 0.0) + float(total.detach().item())
        # 记录各个损失项的原始数值
        for name, value in values.items():
            sums[name] = sums.get(name, 0.0) + value
        # 记录每个损失项当前使用的权重
        for name, weight in weights.items():
            sums[f"weight_{name}"] = sums.get(f"weight_{name}", 0.0) + weight
        # 记录已经处理了多少个 batch
        batch_counts += 1
        # 记录整个训练过程已经更新了多少次模型
        global_step += 1
    # 用所有 batch 的损失总和除以 batch 数量，得到平均损失
    # metrics = {k: v / max(batch_counts, 1) for k, v in sums.items()}
    # 上面等价于下面
    metrics = {}
    # 计算平均值，返回这一轮的平均指标，以及更新后的全局训练步数
    for name, value in sums.items():
        average = value / max(batch_counts, 1)
        metrics[name] = average
    return metrics, global_step

# 验证函数
@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loss_fn: Callable[[dict, dict, int], tuple[Tensor, dict, dict]],
    loader: DataLoader,
    device: torch.device,
    global_step: int,
    cond_keys: tuple[str, ...] = (),
) -> dict[str, float]:
    """検証エポック（勾配計算なし）．
    切换到验证模式"""
    model.eval()
    # 用来累计验证集的损失
    sums: dict[str, float] = {}
    batch_counts = 0
    for batch in loader:
        batch = _move(batch, device)
        # 使用验证数据进行前向计算
        outputs = model(batch["x"])
        # 计算验证机损失。_ 表示不需要第三个返回值 weights
        total, values, _ = loss_fn(outputs, batch, global_step)
        # 计算总损失
        sums["total"] = sums.get("total", 0.0) + float(total.item())
        # 累加重建损失、KL 损失等各项指标
        for name, value in values.items():
            sums[name] = sums.get(name, 0.0) + value
        # 记录已经处理的验证 batch 数
        batch_counts += 1
    # 计算整个验证集的平均损失。并返回
    return {name: value / max(batch_counts, 1) for name, value in sums.items()}
