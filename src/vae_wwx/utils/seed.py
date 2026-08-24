"""乱数シードの一括設定.
固定项目中使用的随机种子。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Python / NumPy / PyTorch のシードをまとめて設定.
    同时固定Python、NumPy和PyTorch的随机种子。"""

    # 固定Python自己的随机数。
    random.seed(seed)
    # 固定NumPy的随机数。
    np.random.seed(seed)
    # 固定Python哈希随机状态。
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 固定PyTorch CPU随机数。
    torch.manual_seed(seed)

    # CUDA可用时固定GPU随机状态。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 尽量使用结果确定的PyTorch算法。
    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )