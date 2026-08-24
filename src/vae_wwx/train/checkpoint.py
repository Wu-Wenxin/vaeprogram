"""チェックポイントの保存と読込．
保存和读取模型训练检查点。"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from vae_wwx.config.schema import ExperimentConfig


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cfg: ExperimentConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    """state_dict + cfg + 任意情報を1ファイルにまとめて保存．
    将模型、优化器和配置保存到一个文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 保存模型参数和实验配置。
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": dataclasses.asdict(cfg),
    }
    # 存在优化器时保存优化器状态。
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    # 存在其他信息时一起保存。
    if extra:
        payload["extra"] = extra
    # 将所有内容保存到文件。
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """``model`` / ``optimizer`` の重みをロードし，残りの payload を返す．
    读取检查点，并恢复模型和优化器。"""
    # 从文件读取检查点内容。
    payload = torch.load(path, map_location=map_location)
    # 恢复模型参数。
    model.load_state_dict(payload["model"])
    # 检查点包含优化器时恢复优化器状态。
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    # 返回完整检查点，方便读取配置等信息。
    return payload
