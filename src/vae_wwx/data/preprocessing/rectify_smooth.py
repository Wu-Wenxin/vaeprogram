"""数据预处理方法：整流平滑
检查二维数据
→ 全波整流
→ 2 Hz低通平滑
→ 转换为float32"""

from __future__ import annotations

import numpy as np
from scipy import signal


# 原始数据采样频率为 2000 Hz。
SAMPLING_FREQUENCY = 2000.0

# Butterworth 低通滤波器阶数。
FILTER_ORDER = 2

# 低通截止频率为 2 Hz，用于提取平滑包络。
CUTOFF_FREQUENCY = 2.0

def rectification(data: np.ndarray) -> np.ndarray:
    """全波整流，把负值转换成正值。"""
    return np.abs(data)

def smoothing(
    data: np.ndarray,
    order: int,
    sampling_freq: float,
    cutoff_freq: float,
) -> np.ndarray:
    """使用 Butterworth 低通滤波器平滑整流后的 EMG。"""
    # 转换成 SciPy 滤波器要求的归一化截止频率。
    normalized_cutoff = cutoff_freq / (sampling_freq / 2)
    # 创建 Butterworth 低通滤波器。
    b, a = signal.butter(order, normalized_cutoff, btype="low")
    # 沿时间轴分别对每个通道进行双向滤波。
    return signal.filtfilt(b, a, data, axis=0,)

def preprocess_rectify_smooth(
    data: np.ndarray,
) -> np.ndarray:
    """当前预处理方案：先全波整流，再低通平滑。"""

    # 确保输入是 float32 的二维数组。
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(
            f"EMG数据必须是二维数组 (T, C)，实际形状为 {data.shape}"
        )

    rectified = rectification(data)
    smoothed = smoothing(
        rectified,
        order=FILTER_ORDER,
        sampling_freq=SAMPLING_FREQUENCY,
        cutoff_freq=CUTOFF_FREQUENCY,
    )

    # 转换成 float32，方便之后转成 PyTorch Tensor。
    return smoothed.astype(np.float32)